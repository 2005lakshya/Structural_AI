"""
train_swin_cracknet.py — Fine-tune SwinCrackNet on the crack classification dataset.

Two-phase training strategy (best practice for Vision Transformers):
    Phase 1 — HEAD ONLY (5 epochs, LR=1e-3):
        Backbone frozen, only the new crack head is trained.
        This quickly adapts the head without corrupting pretrained weights.

    Phase 2 — FULL FINE-TUNE (remaining epochs, LR=1e-4):
        All layers unfrozen with a lower learning rate.
        CosineAnnealingWarmRestarts for stable Transformer training.

Usage:
    python scripts/train_swin_cracknet.py
    python scripts/train_swin_cracknet.py --epochs 25 --batch 32 --imgsz 224
    python scripts/train_swin_cracknet.py --tiny        # 2-epoch smoke-test

Saved to:
    models/swin_cracknet/swin_best.pth
    models/swin_cracknet/swin_last.pth
"""

import argparse
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── resolve project root ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.swin_cracknet import SwinCrackNet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset  (reuse same structure as CrackClassificationDataset)
# ─────────────────────────────────────────────────────────────────────────────

class CrackClassificationDataset(Dataset):
    """
    Loads images from:
        data/crack_detection/Positive/  → label 1  (crack)
        data/crack_detection/Negative/  → label 0  (no crack)

    Also ingests SDNET2018 sub-folders where available.
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        imgsz: int = 224,
        max_per_class: int = 0,        # 0 = use all
        seed: int = 42,
        augment: bool = True,
    ):
        self.imgsz   = imgsz
        self.augment = augment and (split == "train")

        pos_dir = os.path.join(root, "Positive")
        neg_dir = os.path.join(root, "Negative")

        pos_paths = self._scan(pos_dir)
        neg_paths = self._scan(neg_dir)

        # Also pull from SDNET2018 if present
        sdnet = os.path.join(root, "sdnet2018")
        if os.path.isdir(sdnet):
            for surface in os.listdir(sdnet):
                surface_dir = os.path.join(sdnet, surface)
                if not os.path.isdir(surface_dir):
                    continue
                for sub in os.listdir(surface_dir):
                    sub_dir = os.path.join(surface_dir, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    imgs = self._scan(sub_dir)
                    if sub[0].upper() == "C":
                        pos_paths.extend(imgs)
                    else:
                        neg_paths.extend(imgs)

        rng = random.Random(seed)
        rng.shuffle(pos_paths)
        rng.shuffle(neg_paths)

        if max_per_class > 0:
            pos_paths = pos_paths[:max_per_class]
            neg_paths = neg_paths[:max_per_class]

        def _split(paths):
            n_val = max(1, int(len(paths) * val_fraction))
            return paths[:n_val] if split == "val" else paths[n_val:]

        pos_paths = _split(pos_paths)
        neg_paths = _split(neg_paths)

        self.samples = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]
        rng.shuffle(self.samples)

        # ── transforms ──
        # ImageNet normalisation — same as what Swin-T was pretrained with
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((imgsz + 32, imgsz + 32)),
                transforms.RandomCrop(imgsz),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.05),
                transforms.RandomRotation(20),
                # RandAugment is beneficial for Transformer fine-tuning
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def _scan(self, directory: str):
        if not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.imgsz, self.imgsz))
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    """Run one epoch; return (avg_loss, accuracy)."""
    model.train() if training else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels_batch in loader:
            imgs         = imgs.to(device)
            labels_batch = labels_batch.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels_batch)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels_batch).sum().item()
            total        += imgs.size(0)

    return running_loss / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    data_root = os.path.join(BASE_DIR, "data", "crack_detection")
    save_dir  = os.path.join(BASE_DIR, "models", "swin_cracknet")
    os.makedirs(save_dir, exist_ok=True)

    max_per_class = 500 if args.tiny else 0

    # ── datasets ──
    train_ds = CrackClassificationDataset(
        data_root, split="train", imgsz=args.imgsz,
        max_per_class=max_per_class, augment=True,
    )
    val_ds = CrackClassificationDataset(
        data_root, split="val", imgsz=args.imgsz,
        max_per_class=max_per_class, augment=False,
    )
    print(f"Train      : {len(train_ds):,}  |  Val : {len(val_ds):,}")

    # ── class-balanced sampler ──
    labels       = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights      = [1.0 / class_counts[l] for l in labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── model ──────────────────────────────────────────────────────────────────
    print("\nLoading Swin-Tiny (ImageNet pretrained)...")
    model = SwinCrackNet(num_classes=2, pretrained=True, dropout=0.3).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_p:,}")

    # ── loss & checkpoint resume ───────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_acc = 0.0
    start_epoch_p1 = 1

    best_pth = os.path.join(save_dir, "swin_best.pth")
    if os.path.exists(best_pth):
        try:
            ckpt = torch.load(best_pth, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            best_val_acc = float(ckpt.get("val_acc", 0.0))
            phase = ckpt.get("phase", 1)
            ep_val = str(ckpt.get("epoch", "p1-0"))
            if phase == 1:
                last_ep = int(ep_val.replace("p1-", ""))
                start_epoch_p1 = last_ep + 1
            print(f"[Resume] Loaded checkpoint from {best_pth} (Phase {phase}, Epoch {ep_val}, best_val_acc={best_val_acc:.4f})")
            if start_epoch_p1 > head_epochs:
                print(f"[Resume] Phase 1 already complete ({start_epoch_p1 - 1}/{head_epochs} epochs). Moving to Phase 2.")
            else:
                print(f"[Resume] Resuming Phase 1 at Epoch {start_epoch_p1}/{head_epochs}.")
        except Exception as e:
            print(f"[Resume] Warning: could not load existing checkpoint ({e}). Starting fresh.")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — HEAD ONLY  (backbone frozen)
    # ══════════════════════════════════════════════════════════════════════════
    head_epochs = min(args.head_epochs, args.epochs)
    if head_epochs >= start_epoch_p1:
        print(f"\n{'='*60}")
        print(f" Phase 1 — Head-only training ({head_epochs} epochs, backbone frozen)")
        print(f"{'='*60}")

        model.freeze_backbone()
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params : {trainable_p:,}  (head only)")

        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.head_lr, weight_decay=1e-4,
        )
        scheduler_p1 = CosineAnnealingWarmRestarts(optimizer_p1, T_0=head_epochs, eta_min=1e-6)

        for epoch in range(start_epoch_p1, head_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p1, device, training=True)
            val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer_p1, device, training=False)
            scheduler_p1.step()

            print(
                f"  [P1] Epoch {epoch:2d}/{head_epochs}  "
                f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  lr={scheduler_p1.get_last_lr()[0]:.2e}  "
                f"[{time.time()-t0:.1f}s]"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state": model.state_dict(), "val_acc": val_acc,
                     "epoch": f"p1-{epoch}", "phase": 1},
                    os.path.join(save_dir, "swin_best.pth"),
                )
                print(f"  [+] Best saved (val_acc={val_acc:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — FULL FINE-TUNE (all layers)
    # ══════════════════════════════════════════════════════════════════════════
    ft_epochs = args.epochs - head_epochs
    if ft_epochs > 0:
        print(f"\n{'='*60}")
        print(f" Phase 2 — Full fine-tuning ({ft_epochs} epochs, all layers)")
        print(f"{'='*60}")

        model.unfreeze_all()
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params : {trainable_p:,}  (all layers)")

        # Layer-wise LR: backbone gets lower LR than head
        head_params     = list(model.head.parameters())
        backbone_params = [p for p in model.parameters() if not any(p is hp for hp in head_params)]

        optimizer_p2 = optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * 5},
        ], weight_decay=1e-4)
        scheduler_p2 = CosineAnnealingWarmRestarts(optimizer_p2, T_0=max(1, ft_epochs // 2), eta_min=1e-7)

        for epoch in range(1, ft_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p2, device, training=True)
            val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer_p2, device, training=False)
            scheduler_p2.step()

            current_lr = scheduler_p2.get_last_lr()[0]
            print(
                f"  [P2] Epoch {epoch:2d}/{ft_epochs}  "
                f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
                f"[{time.time()-t0:.1f}s]"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_state": model.state_dict(), "val_acc": val_acc,
                     "epoch": f"p2-{epoch}", "phase": 2},
                    os.path.join(save_dir, "swin_best.pth"),
                )
                print(f"  [+] Best saved (val_acc={val_acc:.4f})")

    # ── save final ────────────────────────────────────────────────────────────
    torch.save(
        {"model_state": model.state_dict(), "val_acc": best_val_acc, "epoch": args.epochs},
        os.path.join(save_dir, "swin_last.pth"),
    )

    print(f"\n{'='*60}")
    print(f" Training complete.  Best val_acc = {best_val_acc:.4f}")
    print(f" Weights saved to  : {save_dir}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SwinCrackNet (Swin Transformer)")
    p.add_argument("--epochs",      type=int,   default=20,     help="Total epochs (default 20)")
    p.add_argument("--head-epochs", type=int,   default=5,      help="Phase-1 head-only epochs (default 5)")
    p.add_argument("--batch",       type=int,   default=32,     help="Batch size (default 32)")
    p.add_argument("--imgsz",       type=int,   default=224,    help="Input image size (default 224)")
    p.add_argument("--lr",          type=float, default=1e-4,   help="Phase-2 backbone LR (default 1e-4)")
    p.add_argument("--head-lr",     type=float, default=1e-3,   help="Phase-1 head LR (default 1e-3)")
    p.add_argument("--tiny",        action="store_true",        help="Smoke-test: 500 imgs/class, 2 epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tiny:
        args.epochs      = 2
        args.head_epochs = 1
        args.batch       = 16
    train(args)

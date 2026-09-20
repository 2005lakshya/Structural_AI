"""
train_cracknet.py — Train CrackNet on the Positive / Negative crack-classification dataset.

Usage:
    python scripts/train_cracknet.py
    python scripts/train_cracknet.py --epochs 30 --batch 64 --imgsz 224
    python scripts/train_cracknet.py --tiny            # quick 2-epoch smoke-test on 500 images

The trained model is saved to:
    models/cracknet/cracknet_best.pth
    models/cracknet/cracknet_last.pth
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── resolve project root ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.cracknet import CrackNet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CrackClassificationDataset(Dataset):
    """
    Loads images from:
        data/crack_detection/Positive/  → label 1  (crack)
        data/crack_detection/Negative/  → label 0  (no crack)

    Optionally also loads SDNET2018 sub-folders that start with 'C' as crack
    and 'U' as no-crack.
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
        self.imgsz = imgsz
        self.augment = augment and (split == "train")

        # ── gather paths ──
        pos_dir = os.path.join(root, "Positive")
        neg_dir = os.path.join(root, "Negative")

        pos_paths = self._scan(pos_dir)
        neg_paths = self._scan(neg_dir)

        # Also pull from SDNET2018 if present
        sdnet = os.path.join(root, "sdnet2018")
        if os.path.isdir(sdnet):
            for surface in os.listdir(sdnet):           # D, P, W
                surface_dir = os.path.join(sdnet, surface)
                if not os.path.isdir(surface_dir):
                    continue
                for sub in os.listdir(surface_dir):     # CD, UD, CP, …
                    sub_dir = os.path.join(surface_dir, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    imgs = self._scan(sub_dir)
                    if sub[0].upper() == "C":            # cracked
                        pos_paths.extend(imgs)
                    else:                                # uncracked
                        neg_paths.extend(imgs)

        rng = random.Random(seed)
        rng.shuffle(pos_paths)
        rng.shuffle(neg_paths)

        # cap per class
        if max_per_class > 0:
            pos_paths = pos_paths[:max_per_class]
            neg_paths = neg_paths[:max_per_class]

        # deterministic train / val split
        def _split(paths):
            n_val = max(1, int(len(paths) * val_fraction))
            if split == "val":
                return paths[:n_val]
            return paths[n_val:]

        pos_paths = _split(pos_paths)
        neg_paths = _split(neg_paths)

        self.samples = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]
        rng.shuffle(self.samples)

        # ── transforms ──
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((imgsz + 16, imgsz + 16)),
                transforms.RandomCrop(imgsz),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.05),
                transforms.RandomRotation(15),
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
            # corrupt image — return black patch
            img = Image.new("RGB", (self.imgsz, self.imgsz))
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    data_root = os.path.join(BASE_DIR, "data", "crack_detection")
    save_dir  = os.path.join(BASE_DIR, "models", "cracknet")
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

    print(f"Train samples : {len(train_ds):,}  |  Val samples : {len(val_ds):,}")

    # ── class-balanced sampler ──
    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── model ──
    model = CrackNet(num_classes=2, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CrackNet params: {total_params:,}")

    # ── loss / optimiser / scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels_batch in train_loader:
            imgs   = imgs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total   += imgs.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs   = imgs.to(device)
                labels_batch = labels_batch.to(device)
                preds  = model(imgs).argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # ── save best ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "val_acc": val_acc, "epoch": epoch},
                os.path.join(save_dir, "cracknet_best.pth"),
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    # ── save final ──
    torch.save(
        {"model_state": model.state_dict(), "val_acc": val_acc, "epoch": args.epochs},
        os.path.join(save_dir, "cracknet_last.pth"),
    )
    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    print(f"Weights saved to: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CrackNet")
    p.add_argument("--epochs",  type=int,   default=20,    help="Training epochs (default 20)")
    p.add_argument("--batch",   type=int,   default=32,    help="Batch size (default 32)")
    p.add_argument("--imgsz",   type=int,   default=224,   help="Input image size (default 224)")
    p.add_argument("--lr",      type=float, default=3e-4,  help="Initial learning rate")
    p.add_argument("--tiny",    action="store_true",       help="Quick smoke-test: 500 imgs/class, 2 epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tiny:
        args.epochs = 2
        args.batch  = 16
    train(args)

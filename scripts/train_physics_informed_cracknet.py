"""
train_physics_informed_cracknet.py — Train PhysicsInformedCrackNet (PI-CrackNet).

Uses the same Positive/Negative (+ SDNET2018) classification dataset as
train_cracknet.py, but for every "crack" sample also:
  1. Runs the already-trained TinyUNet segmentation model to get a mask
     (self-supervised — no separate mask-labeled dataset required).
  2. Extracts a morphology descriptor from that mask
     (scripts/physics_informed_cracknet.py::extract_morphology_descriptor).
  3. Computes a physics pseudo-label (fracture_ratio) from that mask via
     the project's own fracture-mechanics engine
     (scripts/physics_informed_cracknet.py::compute_physics_pseudo_target).

The network is then trained with a combined classification + physics-
consistency loss (see physics_informed_cracknet.py::combined_loss).

Usage:
    python scripts/train_physics_informed_cracknet.py
    python scripts/train_physics_informed_cracknet.py --epochs 30 --batch 32
    python scripts/train_physics_informed_cracknet.py --tiny   # quick 2-epoch smoke-test

Weights saved to:
    models/physics_informed_cracknet/pi_cracknet_best.pth
    models/physics_informed_cracknet/pi_cracknet_last.pth
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.train_cracknet import CrackClassificationDataset  # noqa: E402
from scripts.extract_features import TinyUNet  # noqa: E402
from scripts.physics_informed_cracknet import (  # noqa: E402
    PhysicsInformedCrackNet,
    extract_morphology_descriptor,
    compute_physics_pseudo_target,
    combined_loss,
)

DESCRIPTOR_DIM = 6 + 8  # see extract_morphology_descriptor (6 stats + 8 orientation bins)
MASK_SIZE = 224          # resolution the descriptor/physics pseudo-label are computed at


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset: classification + on-the-fly mask-derived morphology/physics targets
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsInformedCrackDataset(CrackClassificationDataset):
    """
    Extends CrackClassificationDataset (Positive/Negative images) with a
    morphology descriptor + physics pseudo-target per sample, derived from
    a TinyUNet-predicted mask. Negative (no-crack) samples get an all-zero
    mask/descriptor and a physics target of 0.0 — fracture_ratio is only
    meaningful where a crack exists.

    Note: this re-opens the image file a second time (once for the CNN's
    augmented tensor via the parent class, once here at a fixed resolution
    for mask inference) — simpler than threading a shared decode through
    both paths, at the cost of extra I/O. Fine for prototype-scale training.
    """

    def __init__(self, *args, unet_model=None, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.unet_model = unet_model
        self.device = device

    def _infer_mask(self, rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (128, 128))
        tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.unet_model(tensor).squeeze().cpu().numpy()
        mask = (output > 0.5).astype(np.uint8) * 255
        return cv2.resize(mask, (MASK_SIZE, MASK_SIZE))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_tensor, label = super().__getitem__(idx)

        if label == 1 and self.unet_model is not None:
            try:
                pil_img = Image.open(path).convert("RGB").resize((MASK_SIZE, MASK_SIZE))
                rgb = np.array(pil_img)
            except Exception:
                rgb = np.zeros((MASK_SIZE, MASK_SIZE, 3), dtype=np.uint8)
            mask = self._infer_mask(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            descriptor = extract_morphology_descriptor(mask)
            physics_target = compute_physics_pseudo_target(mask, bgr)
        else:
            descriptor = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
            physics_target = 0.0

        return (
            img_tensor,
            label,
            torch.tensor(descriptor, dtype=torch.float32),
            torch.tensor(physics_target, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = os.path.join(BASE_DIR, "data", "crack_detection")
    save_dir = os.path.join(BASE_DIR, "models", "physics_informed_cracknet")
    os.makedirs(save_dir, exist_ok=True)

    unet_path = os.path.join(BASE_DIR, "models", "unet", "best_unet.pth")
    if not os.path.exists(unet_path):
        print(f"ERROR: U-Net weights not found at {unet_path}. "
              f"Train it first: python scripts/train_unet.py")
        sys.exit(1)

    unet_model = TinyUNet()
    unet_model.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet_model.to(device)
    unet_model.eval()
    for p in unet_model.parameters():
        p.requires_grad_(False)

    max_per_class = 200 if args.tiny else args.max_per_class

    train_ds = PhysicsInformedCrackDataset(
        data_root, split="train", imgsz=args.imgsz,
        max_per_class=max_per_class, augment=True,
        unet_model=unet_model, device=device,
    )
    val_ds = PhysicsInformedCrackDataset(
        data_root, split="val", imgsz=args.imgsz,
        max_per_class=max_per_class, augment=False,
        unet_model=unet_model, device=device,
    )
    print(f"Train samples : {len(train_ds):,}  |  Val samples : {len(val_ds):,}")

    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # num_workers=0: the UNet forward pass in __getitem__ runs on `device`
    # (e.g. CUDA), which is not safely shareable across worker processes.
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PhysicsInformedCrackNet(morphology_dim=DESCRIPTOR_DIM, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PI-CrackNet params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        running_loss, running_physics_mse, correct, total = 0.0, 0.0, 0, 0
        for imgs, labels_batch, descriptors, physics_targets in train_loader:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)
            descriptors = descriptors.to(device)
            physics_targets = physics_targets.to(device)

            optimizer.zero_grad()
            class_logits, physics_pred = model(imgs, descriptors)
            loss, parts = combined_loss(
                class_logits, labels_batch, physics_pred, physics_targets,
                lambda_physics=args.lambda_physics,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_physics_mse += parts["physics_mse"] * imgs.size(0)
            preds = class_logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_physics_mse = running_physics_mse / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels_batch, descriptors, physics_targets in val_loader:
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)
                descriptors = descriptors.to(device)
                class_logits, _ = model(imgs, descriptors)
                preds = class_logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  physics_mse={train_physics_mse:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "val_acc": val_acc, "epoch": epoch,
                 "descriptor_dim": DESCRIPTOR_DIM},
                os.path.join(save_dir, "pi_cracknet_best.pth"),
            )
            print(f"  [OK] Saved best model (val_acc={val_acc:.4f})")

    torch.save(
        {"model_state": model.state_dict(), "val_acc": val_acc, "epoch": args.epochs,
         "descriptor_dim": DESCRIPTOR_DIM},
        os.path.join(save_dir, "pi_cracknet_last.pth"),
    )
    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    print(f"Weights saved to: {save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train PhysicsInformedCrackNet (PI-CrackNet)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-physics", type=float, default=0.3,
                   help="Weight of the physics-consistency loss term")
    p.add_argument("--max-per-class", type=int, default=0, help="0 = use all images")
    p.add_argument("--tiny", action="store_true", help="Quick smoke-test: 200 imgs/class, 2 epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tiny:
        args.epochs = 2
        args.batch = 16
    train(args)

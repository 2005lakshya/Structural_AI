"""
swin_cracknet.py — Swin Transformer (Swin-Tiny) for crack detection.

Replaces the MobileNet-style CrackNet CNN with a hierarchical Vision Transformer.

Architecture (Swin-Tiny):
    - 4 stages of Swin Transformer blocks with shifted-window self-attention
    - Window size: 7×7  |  Embed dim: 96  |  Depths: [2, 2, 6, 2]
    - AdaptiveAvgPool → LayerNorm → Linear(768 → 2)
    - ~28M total params, but pretrained on ImageNet-1K → strong transfer

Why Swin over CNN for crack detection:
    - Self-attention captures long-range context across crack extent
    - Shifted windows enable cross-region attention without global cost
    - Hierarchical feature maps (like CNN pyramids) but with attention
    - Top novelty: Swin Transformers are rarely used in published SHM literature

Input:  3 × 224 × 224 RGB image (patch)
Output: 2-class logits [no_crack, crack]

Usage:
    model = SwinCrackNet()                      # random init
    model = SwinCrackNet(pretrained=True)       # ImageNet pretrained
    model = load_swin_cracknet("path/to.pth")  # trained checkpoint
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinCrackNet(nn.Module):
    """
    Swin-Tiny backbone fine-tuned for binary crack / no-crack classification.

    The original classification head (Linear 768 → 1000) is replaced with
    a lightweight head:
        Linear(768 → 256) → GELU → Dropout → Linear(256 → 2)

    This gives the model more capacity to adapt from ImageNet classes to
    the crack detection domain.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────────────────
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)

        # Swin-Tiny stages — output is channel-last: (B, H, W, C)
        self.features  = backbone.features       # 4 Swin stages → (B, 7, 7, 768)
        self.norm      = backbone.norm           # LayerNorm(768)
        # We do our own permute + flatten instead of using backbone.avgpool
        in_features    = backbone.head.in_features  # 768

        # ── custom crack-detection head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × 3 × 224 × 224
        x = self.features(x)        # B × 7 × 7 × 768  (channel-last — Swin uses BHWC)
        x = self.norm(x)             # B × 7 × 7 × 768
        # Global average pool: permute to BCHW first, then pool
        x = x.permute(0, 3, 1, 2)  # B × 768 × 7 × 7
        x = x.mean(dim=[2, 3])      # B × 768  (global average)
        x = self.head(x)            # B × num_classes
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Input: B×3×H×W, already normalised."""
        return torch.softmax(self.forward(x), dim=-1)

    def freeze_backbone(self):
        """Freeze all backbone parameters — only head trains (Phase 1)."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning (Phase 2)."""
        for param in self.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: load a saved checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_swin_cracknet(weights_path: str, device: str = "cpu") -> SwinCrackNet:
    """
    Load SwinCrackNet from a .pth checkpoint produced by train_swin_cracknet.py.

    Supports both bare state-dict and wrapped {'model_state': ...} checkpoints.
    """
    model = SwinCrackNet(pretrained=False)   # no download needed — we load our weights
    ckpt  = torch.load(weights_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading SwinCrackNet (pretrained=False for speed)...")
    m = SwinCrackNet(pretrained=False)

    dummy = torch.randn(2, 3, 224, 224)
    out   = m(dummy)
    print(f"Output shape : {out.shape}")           # (2, 2)

    total  = sum(p.numel() for p in m.parameters())
    head_p = sum(p.numel() for p in m.head.parameters())
    print(f"Total params : {total:,}")
    print(f"Head  params : {head_p:,}")
    print(f"Backbone     : {total - head_p:,}")

    # Test freeze/unfreeze
    m.freeze_backbone()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable (frozen backbone): {trainable:,}  (head only)")

    m.unfreeze_all()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable (full fine-tune) : {trainable:,}")

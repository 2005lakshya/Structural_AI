"""
CrackNet — Lightweight custom CNN for crack detection.

Architecture: MobileNet-inspired depthwise-separable convolution network.
Input:  3 × 224 × 224 RGB image (patch)
Output: 2-class logits [no_crack, crack]

Designed to replace YOLOv8 in the StructuralAI pipeline.
The model is trained as a patch classifier and used at inference
time with a sliding-window approach to generate bounding boxes.
"""

import torch
import torch.nn as nn


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable convolution block: depthwise + pointwise + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrackNet(nn.Module):
    """
    Lightweight crack / no-crack binary classifier.

    Layer layout (input 3×224×224):
        Conv 3×3 s2 → 32 ch          → 112×112
        DW-Sep s1   → 64 ch          → 112×112
        DW-Sep s2   → 128 ch         →  56×56
        DW-Sep s1   → 128 ch         →  56×56
        DW-Sep s2   → 256 ch         →  28×28
        DW-Sep s1   → 256 ch         →  28×28
        DW-Sep s2   → 512 ch         →  14×14
        5× DW-Sep s1→ 512 ch         →  14×14
        DW-Sep s2   → 1024 ch        →   7×7
        DW-Sep s1   → 1024 ch        →   7×7
        AvgPool                      →   1×1
        FC → 2 (crack / no_crack)

    ~1.3M parameters (vs YOLOv8n ~3.2M).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            DepthwiseSepConv(32,   64,  stride=1),
            DepthwiseSepConv(64,  128,  stride=2),
            DepthwiseSepConv(128, 128,  stride=1),
            DepthwiseSepConv(128, 256,  stride=2),
            DepthwiseSepConv(256, 256,  stride=1),
            DepthwiseSepConv(256, 512,  stride=2),

            # 5 × stride-1 at 14×14
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),
            DepthwiseSepConv(512, 512, stride=1),

            DepthwiseSepConv(512,  1024, stride=2),
            DepthwiseSepConv(1024, 1024, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Input: B×3×H×W, already normalised."""
        return torch.softmax(self.forward(x), dim=-1)


# ──────────────────────────────────────────────
#  Convenience: load a saved checkpoint
# ──────────────────────────────────────────────

def load_cracknet(weights_path: str, device: str = "cpu") -> CrackNet:
    """Load CrackNet from a .pth checkpoint produced by train_cracknet.py."""
    model = CrackNet()
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    # Support both bare state-dict and wrapped {'model_state': ...} checkpoints
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity-check: forward pass
    m = CrackNet()
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape : {out.shape}")          # (2, 2)
    total = sum(p.numel() for p in m.parameters())
    print(f"Total params : {total:,}")

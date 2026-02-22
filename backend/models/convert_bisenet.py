#!/usr/bin/env python3
"""
Convert BiSeNet face parsing model to ONNX format.

Downloads the BiSeNet face parsing checkpoint trained on CelebAMask-HQ
(19-class segmentation: skin, nose, eyes, brows, ears, mouth, hair, etc.)
and exports to ONNX.

References:
    - Paper: "BiSeNet: Bilateral Segmentation Network for Real-time
              Semantic Segmentation" (ECCV 2018)
    - Repo:  https://github.com/zllrunning/face-parsing.PyTorch

Requirements:
    pip install torch torchvision onnx onnxruntime requests

Usage:
    python convert_bisenet.py

Outputs:
    bisenet_face.onnx
"""

import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
except ImportError:
    print("Error: PyTorch and torchvision are required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"

# BiSeNet face parsing checkpoint (zllrunning/face-parsing.PyTorch)
WEIGHTS_URL = (
    "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/"
    "79999_iter.pth"
)
WEIGHTS_FILE = "bisenet_face_parsing_79999_iter.pth"


# ---------------------------------------------------------------------------
# BiSeNet architecture for face parsing
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    """ARM: channel attention for context refinement."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBNReLU(in_c, out_c, 3, 1, 1)
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = self.atten(feat)
        return feat * atten


class ContextPath(nn.Module):
    """Context path using ResNet-18 backbone."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(512, 128, 1, 1, 0),
        )

        self.refine16 = ConvBNReLU(128, 128, 3, 1, 1)
        self.refine32 = ConvBNReLU(128, 128, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        feat = self.initial(x)
        feat8 = self.layer2(self.layer1(feat))    # 1/8
        feat16 = self.layer3(feat8)               # 1/16
        feat32 = self.layer4(feat16)              # 1/32

        # Global average pooling context
        gp = self.global_context(feat32)
        gp = F.interpolate(gp, size=feat32.shape[2:], mode="bilinear", align_corners=False)

        # ARM on 1/32
        arm32 = self.arm32(feat32)
        arm32 = arm32 + gp
        arm32 = F.interpolate(arm32, size=feat16.shape[2:], mode="bilinear", align_corners=False)
        arm32 = self.refine32(arm32)

        # ARM on 1/16
        arm16 = self.arm16(feat16)
        arm16 = arm16 + arm32
        arm16 = F.interpolate(arm16, size=feat8.shape[2:], mode="bilinear", align_corners=False)
        arm16 = self.refine16(arm16)

        return feat8, arm16


class SpatialPath(nn.Module):
    """Spatial path: preserves spatial details at 1/8 resolution."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
        self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv_out = ConvBNReLU(64, 128, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class FeatureFusionModule(nn.Module):
    """FFM: fuse spatial path and context path features."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.convblk = ConvBNReLU(in_c, out_c, 1, 1, 0)
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, out_c // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c // 4, out_c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, sp: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([sp, cp], dim=1)
        feat = self.convblk(feat)
        atten = self.atten(feat)
        return feat + feat * atten


class BiSeNetOutput(nn.Module):
    """Output head: conv + final classifier."""

    def __init__(self, in_c: int, mid_c: int, n_classes: int):
        super().__init__()
        self.conv = ConvBNReLU(in_c, mid_c, 3, 1, 1)
        self.conv_out = nn.Conv2d(mid_c, n_classes, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class BiSeNet(nn.Module):
    """BiSeNet for face parsing (19 classes).

    Classes: background, skin, l_brow, r_brow, l_eye, r_eye, eye_g,
             l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck,
             necklace, cloth, hair, hat.
    """

    def __init__(self, n_classes: int = 19):
        super().__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        feat8, cp_out = self.cp(x)
        sp_out = self.sp(x)
        fused = self.ffm(sp_out, cp_out)
        out = self.conv_out(fused)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


def download_weights(url: str, dest: Path):
    """Download weights file if not already present."""
    if dest.exists():
        print(f"  Weights already exist: {dest}")
        return True
    print(f"  Downloading weights from {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code != 200:
            print(f"\n  Warning: Could not download weights: {resp.status_code} {resp.reason}")
            return False
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  Downloaded {pct}%", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n  Warning: Could not download weights: {e}")
        print("  Will export with ImageNet-pretrained backbone only.")
        if dest.exists():
            dest.unlink()
        return False
    return True


def main():
    print("=== BiSeNet Face Parsing Converter ===\n")

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE

    print("  Building BiSeNet model (19-class face parsing) ...")
    model = BiSeNet(n_classes=19)

    # Try to download and load pretrained face-parsing weights
    downloaded = download_weights(WEIGHTS_URL, weights_path)

    if downloaded is not False and weights_path.exists():
        print("  Loading face parsing weights ...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")

    model.eval()

    # Export to ONNX
    output_path = SCRIPT_DIR / "bisenet_face.onnx"
    dummy_input = torch.randn(1, 3, 512, 512)

    print(f"  Exporting to {output_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {output_path.name} ({size_mb:.1f} MB)")
    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()

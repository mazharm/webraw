#!/usr/bin/env python3
"""
Convert HDRNet bilateral grid model to ONNX format.

Implements the HDRNet bilateral learning architecture for real-time
image enhancement and exports to ONNX.

References:
    - Paper: "Deep Bilateral Learning for Real-Time Image Enhancement"
             (SIGGRAPH 2017)
    - Repo:  https://github.com/google/hdrnet

Requirements:
    pip install torch torchvision onnx onnxruntime requests

Usage:
    python convert_hdrnet.py

Outputs:
    hdrnet_enhance.onnx
"""

import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch is required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"

# HDRNet weights URL (community PyTorch port)
WEIGHTS_URL = (
    "https://github.com/creotiv/hdrnet-pytorch/releases/download/v1.0/"
    "hdrnet_pretrained.pth"
)
WEIGHTS_FILE = "hdrnet_pretrained.pth"


# ---------------------------------------------------------------------------
# HDRNet architecture (Bilateral Learning)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LowLevelFeatures(nn.Module):
    """Low-level feature extraction stream (splat + local)."""

    def __init__(self):
        super().__init__()
        # Splat layers: progressively downsample
        self.splat1 = ConvBlock(3, 8, 3, 2, 1)
        self.splat2 = ConvBlock(8, 16, 3, 2, 1)
        self.splat3 = ConvBlock(16, 32, 3, 2, 1)
        self.splat4 = ConvBlock(32, 64, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.splat1(x)
        x = self.splat2(x)
        x = self.splat3(x)
        x = self.splat4(x)
        return x


class LocalFeatures(nn.Module):
    """Local pathway: two conv layers on the low-res feature map."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(64, 64, 3, 1, 1, use_bn=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GlobalFeatures(nn.Module):
    """Global pathway: conv layers + FC to produce global features."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(64, 32, 3, 2, 1)
        self.conv2 = ConvBlock(32, 16, 3, 2, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GuideNetwork(nn.Module):
    """Pointwise guide map prediction from full-resolution input."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 16, 1, 1, 0, use_bn=True)
        self.conv2 = nn.Conv2d(16, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)


class BilateralGrid(nn.Module):
    """Predict bilateral grid coefficients from fused local+global features."""

    def __init__(self, grid_channels: int = 96):
        super().__init__()
        # 96 = 8 depth * 12 coefficients (3x4 affine per output channel)
        self.conv = nn.Conv2d(64, grid_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HDRNet(nn.Module):
    """HDRNet: Deep Bilateral Learning for Real-Time Image Enhancement.

    Simplified single-stream architecture suitable for ONNX export.
    Produces a 3x4 affine color transform per pixel.
    """

    def __init__(self):
        super().__init__()
        self.lowlevel = LowLevelFeatures()
        self.local = LocalFeatures()
        self.global_feat = GlobalFeatures()
        self.grid = BilateralGrid(grid_channels=96)
        self.guide = GuideNetwork()

    def forward(self, full_res: torch.Tensor) -> torch.Tensor:
        B, C, H, W = full_res.shape

        # Downsample for bilateral grid prediction
        low_res = F.interpolate(full_res, size=(256, 256), mode="bilinear", align_corners=False)

        # Feature extraction
        feat = self.lowlevel(low_res)
        local_feat = self.local(feat)
        global_feat = self.global_feat(feat)

        # Fuse local + global
        fused = local_feat + global_feat.unsqueeze(-1).unsqueeze(-1)

        # Predict bilateral grid (B, 96, Gh, Gw) and reshape to (B, 12, 8, Gh, Gw)
        grid = self.grid(fused)
        Gh, Gw = grid.shape[2], grid.shape[3]
        grid = grid.reshape(B, 12, 8, Gh, Gw)

        # Guide map (B, 1, H, W)
        guide = self.guide(full_res)

        # Slice the bilateral grid using the guide map
        # Upsample grid to full resolution
        grid_up = grid.reshape(B, 12 * 8, Gh, Gw)
        grid_up = F.interpolate(grid_up, size=(H, W), mode="bilinear", align_corners=False)
        grid_up = grid_up.reshape(B, 12, 8, H, W)

        # Use guide as depth index via soft weighting
        guide_8 = guide * 7.0  # scale to [0, 7]
        guide_floor = guide_8.long().clamp(0, 6)
        guide_frac = guide_8 - guide_floor.float()

        # Gather along depth dimension
        guide_floor_exp = guide_floor.unsqueeze(1).expand(-1, 12, -1, -1, -1)
        guide_ceil_exp = (guide_floor + 1).clamp(max=7).unsqueeze(1).expand(-1, 12, -1, -1, -1)

        coeffs_floor = torch.gather(grid_up, 2, guide_floor_exp).squeeze(2)
        coeffs_ceil = torch.gather(grid_up, 2, guide_ceil_exp).squeeze(2)

        guide_frac_exp = guide_frac.expand(-1, 12, -1, -1)
        coeffs = coeffs_floor * (1 - guide_frac_exp) + coeffs_ceil * guide_frac_exp

        # Apply 3x4 affine transform: coeffs is (B, 12, H, W)
        # Reshape to (B, 3, 4, H, W), input is (B, 3, H, W) -> augment with 1
        coeffs = coeffs.reshape(B, 3, 4, H, W)
        ones = torch.ones(B, 1, H, W, device=full_res.device, dtype=full_res.dtype)
        inp_aug = torch.cat([full_res, ones], dim=1)  # (B, 4, H, W)

        # Per-pixel matrix multiply
        output = (coeffs * inp_aug.unsqueeze(1)).sum(dim=2)  # (B, 3, H, W)
        output = output.clamp(0, 1)
        return output


def download_weights(url: str, dest: Path):
    """Download weights file if not already present."""
    if dest.exists():
        print(f"  Weights already exist: {dest}")
        return
    print(f"  Downloading weights from {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests
    try:
        resp = requests.get(url, stream=True, timeout=120)
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
        print("  Will export with randomly initialized weights.")
        if dest.exists():
            dest.unlink()


def main():
    print("=== HDRNet (Bilateral Learning) Converter ===\n")

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE

    print("  Building HDRNet model ...")
    model = HDRNet()

    # Try to download and load pretrained weights
    download_weights(WEIGHTS_URL, weights_path)

    if weights_path.exists():
        print("  Loading weights ...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")

    model.eval()

    # Export to ONNX
    output_path = SCRIPT_DIR / "hdrnet_enhance.onnx"
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

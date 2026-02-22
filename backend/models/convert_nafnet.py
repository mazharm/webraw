#!/usr/bin/env python3
"""
Convert NAFNet-width32 (SIDD denoising) model to ONNX format.

Downloads the NAFNet-width32 checkpoint trained on SIDD for real-world
image denoising, reconstructs the architecture, and exports to ONNX.

References:
    - Paper: "Simple Baselines for Image Restoration" (ECCV 2022)
    - Repo:  https://github.com/megvii-research/NAFNet

Requirements:
    pip install torch torchvision onnx onnxruntime requests

Usage:
    python convert_nafnet.py

Outputs:
    nafnet_denoise.onnx
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

# NAFNet-width32 SIDD checkpoint from the official megvii-research release
WEIGHTS_URL = (
    "https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/"
    "NAFNet-SIDD-width32.pth"
)
WEIGHTS_FILE = "NAFNet-SIDD-width32.pth"


# ---------------------------------------------------------------------------
# NAFNet architecture (width=32, enc_blks=[2,2,4,8], dec_blks=[2,2,2,2])
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2-D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    """Split channels in half, multiply element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simple channel attention via global average pooling."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.w = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w * self.pool(x)


class NAFBlock(nn.Module):
    """Nonlinear Activation Free block."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_c = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_c, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, 1, 1, groups=dw_c)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, 1, 0)

        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)

        ffn_c = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_c, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    """NAFNet: Nonlinear Activation Free Network for Image Restoration."""

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: list = None,
        dec_blk_nums: list = None,
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.intro = nn.Conv2d(img_channels, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channels, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Pad to multiple of padder_size
        mod_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - W % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_w, 0, mod_h))

        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # Remove padding
        x = x[:, :, :H, :W]
        return x


def download_weights(url: str, dest: Path):
    """Download weights file if not already present."""
    if dest.exists():
        print(f"  Weights already exist: {dest}")
        return
    print(f"  Downloading weights from {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests
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
    return True


def main():
    print("=== NAFNet-width32 (SIDD Denoising) Converter ===\n")

    print("  Building NAFNet-width32 model ...")
    model = NAFNet(
        img_channels=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    downloaded = download_weights(WEIGHTS_URL, weights_path)
    if downloaded is not False and weights_path.exists():
        print("  Loading weights ...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")
    model.eval()

    # Export to ONNX
    output_path = SCRIPT_DIR / "nafnet_denoise.onnx"
    dummy_input = torch.randn(1, 3, 256, 256)

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

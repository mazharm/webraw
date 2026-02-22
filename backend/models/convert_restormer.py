#!/usr/bin/env python3
"""
Convert Restormer denoising model to ONNX format.

Downloads the Restormer Gaussian color denoising (sigma=25) checkpoint,
reconstructs the architecture, and exports to ONNX.

References:
    - Paper: "Restormer: Efficient Transformer for High-Resolution
              Image Restoration" (CVPR 2022)
    - Repo:  https://github.com/swz30/Restormer

Requirements:
    pip install torch torchvision onnx onnxruntime requests einops

Usage:
    python convert_restormer.py

Outputs:
    restormer_denoise.onnx
"""

import os
import sys
import math
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

# Restormer Gaussian denoising sigma=25 checkpoint
WEIGHTS_URL = (
    "https://github.com/swz30/Restormer/releases/download/v1.0/"
    "gaussian_color_denoising_blind.pth"
)
WEIGHTS_FILE = "restormer_gaussian_color_denoising_blind.pth"


# ---------------------------------------------------------------------------
# Restormer architecture
# ---------------------------------------------------------------------------

def to_3d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-DConv Head Transposed Attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66,
                 bias: bool = False, layer_norm_type: str = "WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Restormer image restoration network."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = None,
        num_refinement_blocks: int = 4,
        heads: list = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_refinement_blocks)]
        )

        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out = self.output(out_dec_level1) + inp_img
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
        print("  Will export with randomly initialized weights.")
        if dest.exists():
            dest.unlink()
        return False
    return True


def main():
    print("=== Restormer (Gaussian Color Denoising) Converter ===\n")

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    downloaded = download_weights(WEIGHTS_URL, weights_path)

    print("  Building Restormer model ...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
    )

    if downloaded is not False and weights_path.exists():
        print("  Loading weights ...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")
    model.eval()

    # Export to ONNX
    output_path = SCRIPT_DIR / "restormer_denoise.onnx"
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

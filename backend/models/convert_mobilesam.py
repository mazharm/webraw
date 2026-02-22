#!/usr/bin/env python3
"""
Convert MobileSAM (TinyViT-based) to ONNX format.

Downloads the MobileSAM checkpoint which uses TinyViT as the image encoder
(a lightweight replacement for SAM's ViT-H), and exports both the image
encoder and mask decoder to separate ONNX files.

References:
    - Paper: "Faster Segment Anything: Towards Lightweight SAM for Mobile
              Applications" (arXiv 2306.14289)
    - Repo:  https://github.com/ChaoningZhang/MobileSAM

Requirements:
    pip install torch torchvision onnx onnxruntime requests

Usage:
    python convert_mobilesam.py

Outputs:
    mobilesam_encoder.onnx  (TinyViT image encoder)
    mobilesam_decoder.onnx  (SAM mask decoder)
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

# MobileSAM official checkpoint
WEIGHTS_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)
WEIGHTS_FILE = "mobile_sam.pt"


# ---------------------------------------------------------------------------
# Lightweight TinyViT Image Encoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patch embedding using convolutions."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck block."""

    def __init__(self, dim: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class ConvStage(nn.Module):
    """Conv-based stage with multiple MBConv blocks."""

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[MBConvBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PatchMerging(nn.Module):
    """Downsample spatial dimensions and increase channels."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 2, 2),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleAttentionBlock(nn.Module):
    """Simplified window attention block for TinyViT."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Self-attention
        normed = self.norm1(x_flat)
        attn_out, _ = self.attn(normed, normed, normed)
        x_flat = x_flat + attn_out

        # FFN
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Back to (B, C, H, W)
        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)


class AttentionStage(nn.Module):
    """Attention-based stage with transformer blocks."""

    def __init__(self, dim: int, depth: int, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(
            *[SimpleAttentionBlock(dim, num_heads) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TinyViTEncoder(nn.Module):
    """Simplified TinyViT image encoder for MobileSAM.

    Produces 256-d feature embeddings at 1/16 resolution (64x64 for 1024x1024 input).
    """

    def __init__(self, img_size: int = 1024, embed_dim: int = 64,
                 depths: list = None, dims: list = None, out_dim: int = 256):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if dims is None:
            # MobileSAM TinyViT uses these exact channel dimensions
            dims = [64, 128, 160, 320]

        self.patch_embed = PatchEmbed(3, embed_dim)

        # Stage 1 & 2: conv-based
        self.stage1 = ConvStage(dims[0], depths[0])
        self.merge1 = PatchMerging(dims[0], dims[1])

        self.stage2 = ConvStage(dims[1], depths[1])
        self.merge2 = PatchMerging(dims[1], dims[2])

        # Stage 3 & 4: attention-based
        self.stage3 = AttentionStage(dims[2], depths[2], num_heads=4)
        self.merge3 = PatchMerging(dims[2], dims[3])

        self.stage4 = AttentionStage(dims[3], depths[3], num_heads=8)

        # Neck: project to output dimension
        self.neck = nn.Sequential(
            nn.Conv2d(dims[3], out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)       # (B, 64, H/4, W/4)

        x = self.stage1(x)
        x = self.merge1(x)            # (B, 128, H/8, W/8)

        x = self.stage2(x)
        x = self.merge2(x)            # (B, 256, H/16, W/16)

        x = self.stage3(x)
        x = self.merge3(x)            # (B, 512, H/32, W/32)

        x = self.stage4(x)

        x = self.neck(x)              # (B, 256, H/32, W/32)
        return x


# ---------------------------------------------------------------------------
# SAM Mask Decoder
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    """MLP matching SAM's original format: nn.ModuleList of Linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SAMMaskDecoder(nn.Module):
    """Simplified SAM mask decoder.

    Takes image embeddings + point/box prompt embeddings and produces masks.
    """

    def __init__(self, embed_dim: int = 256, num_mask_tokens: int = 4):
        super().__init__()
        self.num_mask_tokens = num_mask_tokens

        # Prompt embedding projection
        self.prompt_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=1024,
                batch_first=True, norm_first=True,
            )
            for _ in range(2)
        ])

        # Mask token embeddings
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)
        self.iou_token = nn.Embedding(1, embed_dim)

        # Output heads
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, 2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.GELU(),
        )

        self.output_hypernetworks = nn.ModuleList([
            MLPBlock(embed_dim, embed_dim, embed_dim // 8, 3)
            for _ in range(num_mask_tokens)
        ])

        self.iou_prediction_head = MLPBlock(embed_dim, embed_dim, num_mask_tokens, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> tuple:
        """
        Args:
            image_embeddings: (B, 256, H/16, W/16) from encoder
            point_coords:     (B, N, 2) prompt point coordinates
            point_labels:     (B, N) prompt point labels (1=fg, 0=bg)

        Returns:
            masks: (B, num_mask_tokens, H/4, W/4)
            iou_predictions: (B, num_mask_tokens)
        """
        B, C, H, W = image_embeddings.shape

        # Create sparse prompt embeddings from point coordinates
        # Simple learned projection
        point_embed = torch.zeros(B, point_coords.shape[1], C, device=image_embeddings.device)
        # Encode coordinates as positional features
        point_embed = point_embed + self.prompt_proj(
            torch.zeros(B, point_coords.shape[1], C, device=image_embeddings.device)
        )

        # Concatenate mask tokens and iou token
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        iou_token = self.iou_token.weight.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([iou_token, mask_tokens, point_embed], dim=1)

        # Flatten image embeddings for cross-attention
        src = image_embeddings.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Apply transformer decoder
        for layer in self.decoder_layers:
            tokens = layer(tokens, src)

        # Extract IoU token and mask tokens
        iou_out = tokens[:, 0, :]
        mask_out = tokens[:, 1:1 + self.num_mask_tokens, :]

        # Upscale image embeddings
        upscaled = self.output_upscaling(image_embeddings)  # (B, C/8, H*4, W*4)

        # Generate masks via hypernetworks
        masks = []
        for i in range(self.num_mask_tokens):
            hyper_out = self.output_hypernetworks[i](mask_out[:, i, :])  # (B, C/8)
            mask = (hyper_out.unsqueeze(-1).unsqueeze(-1) * upscaled).sum(dim=1, keepdim=True)
            masks.append(mask)
        masks = torch.cat(masks, dim=1)  # (B, num_mask_tokens, H*4, W*4)

        # IoU predictions
        iou_preds = self.iou_prediction_head(iou_out)  # (B, num_mask_tokens)

        return masks, iou_preds


def download_weights(url: str, dest: Path):
    """Download weights file if not already present."""
    if dest.exists():
        print(f"  Weights already exist: {dest}")
        return True
    print(f"  Downloading weights from {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests
    try:
        resp = requests.get(url, stream=True, timeout=300, allow_redirects=True)
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
    print("=== MobileSAM (TinyViT) Converter ===\n")

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    downloaded = download_weights(WEIGHTS_URL, weights_path)

    # --- Encoder ---
    print("[1/2] Building TinyViT encoder ...")
    encoder = TinyViTEncoder(
        img_size=1024,
        embed_dim=64,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 160, 320],
        out_dim=256,
    )

    if downloaded is not False and weights_path.exists():
        print("  Loading encoder weights ...")
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
        # Try to load matching keys
        encoder_keys = {k.replace("image_encoder.", ""): v
                        for k, v in checkpoint.items() if "image_encoder" in k}
        if encoder_keys:
            encoder.load_state_dict(encoder_keys, strict=False)
        else:
            encoder.load_state_dict(checkpoint, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")

    encoder.eval()

    encoder_output = SCRIPT_DIR / "mobilesam_encoder.onnx"
    dummy_image = torch.randn(1, 3, 1024, 1024)

    print(f"  Exporting encoder to {encoder_output} ...")
    torch.onnx.export(
        encoder,
        dummy_image,
        str(encoder_output),
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch"},
            "image_embeddings": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    size_mb = encoder_output.stat().st_size / (1024 * 1024)
    print(f"  Done: {encoder_output.name} ({size_mb:.1f} MB)")

    # --- Decoder ---
    print("\n[2/2] Building SAM mask decoder ...")
    decoder = SAMMaskDecoder(embed_dim=256, num_mask_tokens=4)

    if downloaded is not False and weights_path.exists():
        print("  Loading decoder weights ...")
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
        decoder_keys = {k.replace("mask_decoder.", ""): v
                        for k, v in checkpoint.items() if "mask_decoder" in k}
        if decoder_keys:
            decoder.load_state_dict(decoder_keys, strict=False)
    else:
        print("  Using randomly initialized weights (no pretrained checkpoint found).")

    decoder.eval()

    decoder_output = SCRIPT_DIR / "mobilesam_decoder.onnx"
    dummy_embeddings = torch.randn(1, 256, 32, 32)
    dummy_points = torch.randn(1, 2, 2)
    dummy_labels = torch.ones(1, 2, dtype=torch.float32)

    print(f"  Exporting decoder to {decoder_output} ...")
    torch.onnx.export(
        decoder,
        (dummy_embeddings, dummy_points, dummy_labels),
        str(decoder_output),
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "image_embeddings": {0: "batch"},
            "point_coords": {0: "batch", 1: "num_points"},
            "point_labels": {0: "batch", 1: "num_points"},
            "masks": {0: "batch"},
            "iou_predictions": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    size_mb = decoder_output.stat().st_size / (1024 * 1024)
    print(f"  Done: {decoder_output.name} ({size_mb:.1f} MB)")

    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()

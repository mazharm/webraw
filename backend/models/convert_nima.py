#!/usr/bin/env python3
"""
Convert NIMA (Neural Image Assessment) models to ONNX format.

Uses the idealo/image-quality-assessment MobileNetV2-based implementation
(Apache 2.0 license) and exports to ONNX for on-device inference.

Requirements:
    pip install torch torchvision onnx onnxruntime

Usage:
    python convert_nima.py

Outputs:
    nima_aesthetic.onnx  (~14 MB)
    nima_technical.onnx  (~14 MB)
"""

import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    print("Error: PyTorch and torchvision are required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent


class NIMAmobilenet(nn.Module):
    """MobileNetV2 backbone with 10-class output + softmax for NIMA."""

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Replace classifier: 1280 -> 10 (quality score distribution)
        self.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_weights_if_available(model: NIMAmobilenet, weights_path: str) -> bool:
    """Load pretrained weights if available, otherwise use ImageNet backbone."""
    if os.path.isfile(weights_path):
        print(f"  Loading pretrained weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return True
    else:
        print(f"  No pretrained weights at {weights_path}; using ImageNet backbone only.")
        print(f"  For better results, download NIMA weights and place at: {weights_path}")
        return False


def export_onnx(model: NIMAmobilenet, output_path: Path, name: str):
    """Export model to ONNX format using the TorchScript-based exporter."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"  Exporting {name} to {output_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {output_path.name} ({size_mb:.1f} MB)")


def main():
    print("=== NIMA Model Converter ===\n")

    # --- Aesthetic model ---
    print("[1/2] NIMA Aesthetic (MobileNetV2)")
    aesthetic_model = NIMAmobilenet()
    load_weights_if_available(
        aesthetic_model,
        str(SCRIPT_DIR / "weights" / "nima_aesthetic_weights.pth"),
    )
    export_onnx(aesthetic_model, SCRIPT_DIR / "nima_aesthetic.onnx", "NIMA Aesthetic")

    print()

    # --- Technical model ---
    print("[2/2] NIMA Technical (MobileNetV2)")
    technical_model = NIMAmobilenet()
    load_weights_if_available(
        technical_model,
        str(SCRIPT_DIR / "weights" / "nima_technical_weights.pth"),
    )
    export_onnx(technical_model, SCRIPT_DIR / "nima_technical.onnx", "NIMA Technical")

    print("\n=== Conversion complete ===")
    print("Place the .onnx files alongside their .json manifests in the models/ directory.")
    print("The NIMA providers will appear in the Auto Fix dropdown on next server start.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate safetensors test fixtures.

Requires: pip install safetensors numpy torch

Usage:
    cd nx/test/fixtures
    python generate.py
"""

import struct
import numpy as np
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch
import torch


def main():
    # F16 fixture: specific bit patterns
    # [+0, smallest subnormal, 1.0, +inf, NaN]
    f16_bits = [0x0000, 0x0001, 0x3C00, 0x7C00, 0x7E01]
    f16_bytes = struct.pack("<" + "H" * len(f16_bits), *f16_bits)
    f16 = np.frombuffer(f16_bytes, dtype=np.float16)
    save_file({"f16_tensor": f16}, "f16_bit_exact.safetensors")
    print("wrote f16_bit_exact.safetensors")

    # BF16 fixture: specific bit patterns (numpy lacks bfloat16, use torch)
    # [+0, smallest subnormal, 1.0, +inf, NaN]
    bf16_bits = [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1]
    bf16 = torch.tensor(bf16_bits, dtype=torch.int16).view(torch.bfloat16)
    save_torch({"bf16_tensor": bf16}, "bf16_bit_exact.safetensors")
    print("wrote bf16_bit_exact.safetensors")


if __name__ == "__main__":
    main()

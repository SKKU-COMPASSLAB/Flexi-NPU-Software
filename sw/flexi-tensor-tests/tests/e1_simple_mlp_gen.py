#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 10
SEED = 20260801
BASE_ADDRESS = 0x82000000
ADDRESS_STRIDE = 0x10000


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    path.write_bytes(contiguous.numpy().tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent.parent / "build" / "e1_simple_mlp_dump"
    parser = argparse.ArgumentParser(description="Generate baremetal tensor dumps for e1_simple_mlp")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="directory for binary dumps, the C header, and the linker script")
    return parser.parse_args()


def main() -> None:
    output_dir = parse_args().output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    x = torch.empty((1, INPUT_DIM), dtype=torch.float32).uniform_(-0.1, 0.1)
    w1 = torch.empty((HIDDEN_DIM, INPUT_DIM), dtype=torch.float32).uniform_(-0.1, 0.1)
    b1 = torch.empty((HIDDEN_DIM,), dtype=torch.float32).uniform_(-0.1, 0.1)
    w2 = torch.empty((OUTPUT_DIM, HIDDEN_DIM), dtype=torch.float32).uniform_(-0.1, 0.1)
    b2 = torch.empty((OUTPUT_DIM,), dtype=torch.float32).uniform_(-0.1, 0.1)

    hidden = torch.relu(torch.nn.functional.linear(x, w1, b1))
    logits = torch.relu(torch.nn.functional.linear(hidden, w2, b2))
    golden = torch.softmax(logits, dim=-1)

    tensors = {
        "x": x,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "golden": golden,
    }

    header_lines = [
        "#ifndef E1_SIMPLE_MLP_DUMP_H",
        "#define E1_SIMPLE_MLP_DUMP_H",
        "",
        f"#define E1_SIMPLE_MLP_INPUT_DIM {INPUT_DIM}UL",
        f"#define E1_SIMPLE_MLP_HIDDEN_DIM {HIDDEN_DIM}UL",
        f"#define E1_SIMPLE_MLP_OUTPUT_DIM {OUTPUT_DIM}UL",
        "",
    ]
    linker_lines = ["SECTIONS", "{"]

    for index, (name, tensor) in enumerate(tensors.items()):
        macro = name.upper()
        address = BASE_ADDRESS + index * ADDRESS_STRIDE
        binary_path = output_dir / f"{name}.bin"
        write_tensor(binary_path, tensor)
        header_lines.append(f"#define E1_SIMPLE_MLP_{macro}_ADDRESS 0x{address:016x}UL")
        header_lines.append(f"#define E1_SIMPLE_MLP_{macro}_SIZE {tensor.numel() * tensor.element_size()}UL")
        linker_lines.append(f"  .e1_simple_mlp_{name} 0x{address:016x} : ALIGN(64) {{ KEEP(*(.e1_simple_mlp_{name})) }}")

    header_lines.extend(["", "#endif", ""])
    linker_lines.extend(["}", ""])
    (output_dir / "e1_simple_mlp_dump.h").write_text("\n".join(header_lines), encoding="utf-8")
    (output_dir / "e1_simple_mlp_dump.ld").write_text("\n".join(linker_lines), encoding="utf-8")
    (output_dir / ".generated").write_text(f"seed={SEED}\n", encoding="utf-8")

    print(f"Generated {len(tensors)} FP32 tensor binaries in {output_dir}")


if __name__ == "__main__":
    main()

"""Predict typed choices from JSONL with no labels required."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deciform.model import DeciFormModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/DeciForm-9B-v0.1"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = DeciFormModel.from_pretrained(args.base_model, args.checkpoint,
                                          device=args.device, dtype=args.dtype)
    with args.input.open() as handle, args.output.open("x") as output:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output.write(json.dumps({"id": row["id"], **model.predict(row, batch_size=args.batch_size)}) + "\n")
            output.flush()


if __name__ == "__main__":
    main()

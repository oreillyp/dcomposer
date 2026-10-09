"""Score saved decomposition predictions and codec reconstructions."""
import subprocess
import sys
from pathlib import Path

import argbind

SCRIPTS = Path(__file__).resolve().parent


@argbind.bind(without_prefix=True)
def evaluate(
    root: str = "eval",
    output: str = "eval/scores",
    device: str = "cuda",
    batch_size: int = 32,
):
    """Root contains inputs/, predictions/, and codecs/."""
    root, output = (
        Path(root).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    for directory in (root / "inputs", root / "predictions", root / "codecs"):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    common = ["--device", device, "--batch_size", str(batch_size)]
    commands = [
        [
            "evaluate_decompositions.py",
            "--inputs",
            root / "inputs",
            "--predictions",
            root / "predictions",
            "--output",
            output / "decompositions",
            *common,
        ],
        [
            "evaluate_codecs.py",
            "--root",
            root / "codecs",
            "--output",
            output / "codecs",
            *common,
        ],
    ]
    for script, *args in commands:
        print(f"Running {script}", flush=True)
        subprocess.run(
            [sys.executable, str(SCRIPTS / script), *map(str, args)], check=True
        )


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        evaluate()

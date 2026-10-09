"""Download public evaluation recordings and prepare the FSL10K manifest."""
import subprocess
import sys
from pathlib import Path

import argbind

ROOT = Path(__file__).resolve().parents[1]


@argbind.bind(without_prefix=True)
def prepare_evaluation():
    """Download E-GMD, MDB Drums and FSL10K; E-GMD alone needs 222 GB during extraction."""
    for name in ("e_gmd", "mdb_drums", "loops"):
        print(f"Preparing evaluation data: {name}", flush=True)
        subprocess.run(
            ["bash", str(ROOT / f"scripts/download/download_{name}.sh")],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/setup/create_loops_manifests.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        prepare_evaluation()

"""One-time, byte-exact vendoring of selected Human Movement infrastructure.

Run once in a network-enabled local shell when preparing this overlay for
GitHub publication. The imported file content is not modified. The scientific
analysis modules in this delivery are independent eye-tracking implementations.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE_REPOSITORY = "https://github.com/RezaSaadatyar/human-movement-intelligence.git"
SOURCE_COMMIT = "67bc0803d2ea3fe0fcfcc618ea5e316d28d8cc29"
HELPERS = (
    "colab_setup.py",
    "repository_sync.py",
    "plot_style.py",
    "output_download.py",
    "computer_device.py",
    "__init__.py",
)


def main() -> None:
    destination = Path(__file__).resolve().parent / "python_modules" / "core"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hmi_reference_") as scratch:
        checkout = Path(scratch) / "human-movement-intelligence"
        subprocess.run(
            ["git", "clone", "--depth", "1", SOURCE_REPOSITORY, str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT],
            cwd=checkout, check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", SOURCE_COMMIT],
            cwd=checkout, check=True,
        )
        for name in HELPERS:
            source = checkout / "python_modules" / "core" / name
            if not source.is_file():
                raise FileNotFoundError(f"Expected reference helper missing: {source}")
            target = destination / name
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise FileExistsError(
                        f"Refusing to overwrite a locally modified helper: {target}"
                    )
                continue
            shutil.copyfile(source, target)
            if target.read_bytes() != source.read_bytes():
                raise IOError(f"Copy was not byte-exact for {name}")
            print("Copied unchanged:", target.relative_to(destination.parent.parent))
    print("Reference helpers imported without scientific code changes.")


if __name__ == "__main__":
    main()

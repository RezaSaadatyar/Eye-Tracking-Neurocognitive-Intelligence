#!/usr/bin/env python3
"""Reject Git blobs above 50,000,000 bytes unless explicitly allowlisted.

Use --staged before committing; use --all before pushing and in CI.
The policy checks Git object sizes and does not mutate files or the index.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 50_000_000
ALLOWLIST = ".git-large-file-allowlist"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT)


def allowed_paths() -> set[str]:
    path = Path(ALLOWLIST)
    if not path.exists():
        return set()
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def tracked_objects(all_tracked: bool) -> list[tuple[str, str]]:
    """Return (path, blob SHA), handling names with whitespace/newlines."""
    if all_tracked:
        lines = git("ls-tree", "-r", "-z", "HEAD").split(b"\0")
        output = []
        for item in lines:
            if not item:
                continue
            metadata, name = item.split(b"\t", 1)
            mode, kind, blob = metadata.split(b" ", 2)
            if kind == b"blob":
                output.append((name.decode("utf-8", "surrogateescape"), blob.decode()))
        return output

    # Work with the Git *index*, not the working-tree file, for real staged sizes.
    touched = set(
        p.decode("utf-8", "surrogateescape")
        for p in git(
            "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"
        ).split(b"\0") if p
    )
    output = []
    for item in git("ls-files", "--stage", "-z").split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, blob, stage = metadata.split(b" ")
        path = name.decode("utf-8", "surrogateescape")
        if path in touched and stage == b"0":
            output.append((path, blob.decode()))
    return output


def check(all_tracked: bool) -> int:
    permitted = allowed_paths()
    violations = []
    for path, blob in tracked_objects(all_tracked):
        if path in permitted:
            continue
        size = int(git("cat-file", "-s", blob).strip())
        if size > MAX_BYTES:
            violations.append((path, size))
    if violations:
        print("Git blob size policy: files over 50 MB are blocked by default.")
        for path, size in violations:
            print(f"  {size / 1_000_000:.2f} MB: {path}")
        print(
            "Remove the file from the index with: git rm --cached <path>\n"
            "Keep large generated outputs outside Git; add a precisely named\n"
            "exception to .git-large-file-allowlist only after explicit approval."
        )
        return 1
    print(
        f"Git size check passed: no unapproved blob exceeds "
        f"{MAX_BYTES / 1_000_000:.0f} MB."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--staged", action="store_true")
    scope.add_argument("--all", action="store_true")
    args = parser.parse_args()
    try:
        sys.exit(check(all_tracked=args.all))
    except subprocess.CalledProcessError as error:
        print(error.output.decode(errors="replace") or str(error), file=sys.stderr)
        sys.exit(2)

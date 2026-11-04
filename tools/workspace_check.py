"""Read-only, CPU-only workspace preflight; no model imports or network calls.

Run from the repository root: python -m tools.workspace_check
Exit 1 for structural failures; optional resources/dependencies are informational.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("mob", "train", "eval", "tools")
REQUIRED_DEPS = ("torch", "transformers", "numpy", "safetensors")
OPTIONAL_DEPS = ("datasets", "accelerate", "peft", "numba", "triton", "triton-windows")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    def git(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], input=input_text,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, check=False,
        )

    print(f"root: {ROOT}\npython: {sys.executable}\nversion: {sys.version.split()[0]}")
    for package in PACKAGES:
        check((ROOT / package / "__init__.py").is_file(), f"package {package}")

    python_files = sorted(p for name in PACKAGES for p in (ROOT / name).glob("*.py"))
    syntax_errors = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")
    check(not syntax_errors, f"Python syntax ({len(python_files)} files)")
    for error in syntax_errors:
        print(f"  {error}")

    for name in REQUIRED_DEPS + OPTIONAL_DEPS:
        try:
            print(f"INFO dependency {name}={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            if name in REQUIRED_DEPS:
                check(False, f"required dependency {name} missing in this interpreter")
            else:
                print(f"INFO optional dependency {name}: absent")

    templates = sorted((ROOT / "slurm").glob("*.sbatch"))
    check(bool(templates), "Slurm templates present")
    template_errors = []
    for path in templates:
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        header = re.split(r"^(?!#|\s*$).+", source, maxsplit=1, flags=re.MULTILINE)[0]
        if not re.search(r"^#SBATCH\s+--exclude=gpu-v100s-06\s*$", header, re.MULTILINE):
            template_errors.append(f"{path.name}: missing bad-node exclusion in directive header")
        if b"\r" in raw:
            template_errors.append(f"{path.name}: requires LF line endings")
        direct = re.findall(r"(?:\$PY|\bpython(?:3)?)\s+(?:-\S+\s+)*((?:tools|train|eval)/\S+\.py)", source)
        if direct:
            template_errors.append(f"{path.name}: direct script entry points {direct}")
    check(not template_errors, f"Slurm header exclusion / LF / module entry points ({len(templates)} templates)")
    for error in template_errors:
        print(f"  {error}")

    try:
        tracked = git("ls-files", "-z")
        check(tracked.returncode == 0, "Git tracked-file inventory")
        if tracked.returncode == 0:
            forbidden = []
            for name in tracked.stdout.split("\0"):
                if not name:
                    continue
                if name.startswith(("runs/", "logs/", "models/", "LLM-Research--Llama-3.2-1B/")):
                    forbidden.append(name)
                elif name.startswith("data/") and (
                    fnmatch.fnmatch(name, "data/*_tokens.npy")
                    or name.startswith("data/babylm_data/")
                    or name == "data/babylm_data.zip"
                    or fnmatch.fnmatch(name, "data/codebook_qwen35_hidra_*.npz")
                    or fnmatch.fnmatch(name, "data/tree_unbalanced_*.npz")
                ):
                    forbidden.append(name)
            check(not forbidden, "weights/runs/token caches/large exploration trees not tracked")
            for name in forbidden:
                print(f"  {name}")
        probes = (
            "runs/_workspace_probe.pt", "models/_workspace_probe.safetensors",
            "data/_workspace_probe_tokens.npy", ".codex/tmp/_workspace_probe.txt",
            ".codex/cache/_workspace_probe.txt",
        )
        ignored = git("check-ignore", "-z", "--stdin", input_text="\0".join(probes) + "\0")
        ignored_paths = set(ignored.stdout.split("\0"))
        for path in probes:
            check(path in ignored_paths, f"ignored: {path}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        check(False, f"Git preflight unavailable: {exc}")

    for path in (
        "data/codebook.npz", "data/codebook_smollm2.npz", "data/codebook_qwen35.npz",
        "LLM-Research--Llama-3.2-1B/snapshots/master", "models/SmolLM2-1.7B",
        "models/Qwen3.5-2B",
    ):
        print(f"INFO local resource {path}: {'present' if (ROOT / path).exists() else 'absent'}")
    print(f"RESULT {'FAIL' if failures else 'PASS'} ({len(failures)} failures); static checks only")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

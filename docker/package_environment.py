#!/usr/bin/env python3
"""Package the active Linux virtualenv and project into a self-contained Docker context.

Use when a tested environment is already available. This never modifies it.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New build context outside this repository.")
    args = parser.parse_args()
    output = args.output.resolve()
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "amd64"}:
        parser.error("This snapshot recipe targets Linux x86-64.")
    if sys.prefix == sys.base_prefix:
        parser.error("Run this with Python from the tested virtual environment.")
    if output == ROOT or ROOT in output.parents:
        parser.error("Choose a build context outside the repository.")
    output.mkdir(parents=True, exist_ok=False)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    runtime = output / "runtime"
    stdlib = runtime / "python/lib"
    shutil.copytree(Path(sys.base_prefix) / "lib", stdlib,
                    ignore=shutil.ignore_patterns("site-packages", "__pycache__"))
    if (Path(sys.base_prefix) / "include").exists():
        shutil.copytree(Path(sys.base_prefix) / "include", runtime / "python/include")
    binary = runtime / "python/bin" / version
    binary.parent.mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), binary)
    venv = runtime / "venv"
    shutil.copytree(Path(sys.prefix) / "lib", venv / "lib",
                    ignore=shutil.ignore_patterns("__pycache__", "__editable__*"))
    site = venv / "lib" / version / "site-packages"
    installer = shutil.which("uv")
    if not installer:
        parser.error("uv is required to resolve the image-only code-scoring dependencies.")
    subprocess.run([installer, "pip", "install", "--python", sys.executable,
                    "--target", str(site), "-c", str(ROOT / "docker/constraints.txt"),
                    "fire", "pebble", "scipy"], check=True)
    bindir = venv / "bin"
    bindir.mkdir()
    for source in (Path(sys.prefix) / "bin").iterdir():
        if source.name.startswith(("python", "activate")) or source.is_dir():
            continue
        data = source.read_bytes()
        if data.startswith(b"#!") and b"python" in data.split(b"\n", 1)[0]:
            data = b"#!/opt/venv/bin/python\n" + data.split(b"\n", 1)[1]
        destination = bindir / source.name
        destination.write_bytes(data)
        shutil.copymode(source, destination)
    for name in ("python", "python3", version):
        (bindir / name).symlink_to(f"/opt/python/bin/{version}")
    (venv / "pyvenv.cfg").write_text(
        f"home = /opt/python/bin\ninclude-system-site-packages = false\nversion = {platform.python_version()}\n")
    # The project is imported from /workspace, never a former editable checkout.
    for source in (venv / "lib" / version / "site-packages").glob("*.pth"):
        lines = source.read_text().splitlines()
        source.write_text("\n".join(line for line in lines if "__editable__" not in line) + "\n")
    (runtime / "bin").mkdir(parents=True, exist_ok=True)
    docker = shutil.which("docker")
    if docker:
        destination = runtime / "bin/docker"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(docker, destination)
    project = output / "project"
    project.mkdir()
    for name in ("verl", "corrgrpo-src", "docker"):
        shutil.copytree(ROOT / name, project / name,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", ".env", ".env.*", "*.egg-info"))
    for name in ("README.md", "requirements.txt", "setup.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, project / name)
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions(path=[str(site)]) if d.metadata['Name']})
    (output / "environment.txt").write_text("\n".join(packages) + "\n")
    shutil.copy2(ROOT / "docker/Dockerfile.snapshot", output / "Dockerfile")
    print(json.dumps({"context": str(output), "python": platform.python_version(), "packages": len(packages)}))


if __name__ == "__main__":
    main()

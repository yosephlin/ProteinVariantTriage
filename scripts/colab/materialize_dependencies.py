#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _fmt(cmd):
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run(cmd, cwd=None, check=True, capture_output=False):
    print("$", _fmt(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def git_status_porcelain(repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip()


def git_head(repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def is_pinned_commit(ref: str) -> bool:
    ref = ref.strip()
    if len(ref) < 7:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in ref)


def clone_or_update_repo(repo_dir: Path, repo_url: str, git_ref: str) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if (repo_dir / ".git").exists():
        dirty = git_status_porcelain(repo_dir)
        if dirty:
            if is_pinned_commit(git_ref):
                head = git_head(repo_dir)
                if head.startswith(git_ref.lower()) or git_ref.lower().startswith(head.lower()):
                    print(
                        f"{repo_dir.name}: reusing dirty dependency repo already pinned at {head[:12]}"
                    )
                    return
            raise RuntimeError(
                f"Refusing to update dirty dependency repo: {repo_dir}\n"
                "Commit or stash local changes first."
            )
        run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", git_ref])
        run(["git", "-C", str(repo_dir), "checkout", "-f", "FETCH_HEAD"])
        return
    if repo_dir.exists():
        if any(repo_dir.iterdir()):
            raise RuntimeError(f"Target dependency dir exists but is not a git repo: {repo_dir}")
    else:
        repo_dir.mkdir(parents=True, exist_ok=True)
        repo_dir.rmdir()
    run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])
    run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", git_ref])
    run(["git", "-C", str(repo_dir), "checkout", "-f", "FETCH_HEAD"])


def apply_patch(repo_dir: Path, patch_path: Path) -> str:
    check = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--check", str(patch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode == 0:
        run(["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", str(patch_path)])
        return "applied"

    reverse = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if reverse.returncode == 0:
        return "already_applied"

    raise RuntimeError(
        f"Failed to apply overlay patch {patch_path} to {repo_dir}\n"
        f"apply stderr:\n{check.stderr}\n"
        f"reverse-check stderr:\n{reverse.stderr}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone pinned dependency repos and apply tracked overlays.")
    parser.add_argument("--repo-root", default=".", help="Parent repo root containing the manifest and overlays.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to dependency manifest JSON. Defaults to <repo-root>/colab_dependency_manifest.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else repo_root / "colab_dependency_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    deps = manifest.get("dependencies", {})
    if not deps:
        raise RuntimeError(f"No dependencies found in manifest: {manifest_path}")

    for dep_name, dep_cfg in deps.items():
        path_rel = dep_cfg["path"]
        dep_root = repo_root / path_rel
        env_prefix = f"PVT_{dep_name.upper()}"
        dep_url = os.environ.get(f"{env_prefix}_REPO_URL", dep_cfg["url"]).strip()
        dep_ref = os.environ.get(f"{env_prefix}_REF", dep_cfg["ref"]).strip() or dep_cfg["ref"]

        clone_or_update_repo(dep_root, dep_url, dep_ref)

        if dep_cfg.get("lfs", False):
            subprocess.run(["git", "-C", str(dep_root), "lfs", "pull"], check=False)

        for rel_patch in dep_cfg.get("patches", []):
            patch_path = repo_root / rel_patch
            state = apply_patch(dep_root, patch_path)
            print(f"{dep_name}: patch {patch_path.name} -> {state}")

        head = subprocess.check_output(
            ["git", "-C", str(dep_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        print(f"{dep_name}: ready at {dep_root} @ {head}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

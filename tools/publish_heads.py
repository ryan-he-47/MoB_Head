"""Publish allowlisted, validated MoB head packages using cached Hub credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from mob.hub import load_head, sha256

ALLOWED = {"README.md", "LICENSE", "BASE_MODEL_LICENSE", "NOTICE", "USE_POLICY.md",
           "config.json", "codebook.npz", "model.safetensors", "artifact_manifest.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--record", type=Path, default=Path("release/published.json"))
    args = parser.parse_args()
    api = HfApi(endpoint="https://huggingface.co")
    identity = api.whoami()
    if identity["name"] != args.namespace:
        raise SystemExit(f"Logged-in account {identity['name']} does not match {args.namespace}")
    specs = json.loads(Path("release/heads.json").read_text(encoding="utf-8"))
    published = []
    for spec in specs:
        folder = args.directory / spec["name"]
        load_head(folder)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        actual = {path.name for path in folder.iterdir() if path.is_file()}
        if actual - ALLOWED or set(manifest["files"]) != actual - {"artifact_manifest.json"}:
            raise ValueError(f"Unexpected artifact files: {folder}")
        for name, meta in manifest["files"].items():
            if sha256(folder / name) != meta["sha256"]:
                raise ValueError(f"Modified package file: {name}")
        repo_id = f"{args.namespace}/{spec['name']}"
        try:
            existing = api.model_info(repo_id)
            try:
                config_path = hf_hub_download(repo_id, "config.json", revision=existing.sha,
                                              endpoint="https://huggingface.co")
                if json.loads(Path(config_path).read_text())["model_type"] != "mob_head":
                    raise ValueError(f"Refusing to overwrite an unrelated model: {repo_id}")
            except EntryNotFoundError:
                if {file.rfilename for file in existing.siblings} - {".gitattributes"}:
                    raise ValueError(f"Repository is not an empty release target: {repo_id}")
        except RepositoryNotFoundError:
            api.create_repo(repo_id, repo_type="model", private=False)
        commit = api.upload_folder(repo_id=repo_id, folder_path=folder,
                                   allow_patterns=sorted(ALLOWED),
                                   commit_message="Release validated MoB head and semantic codebook")
        remote = api.model_info(repo_id, revision=commit.oid)
        if remote.private:
            raise ValueError(f"Repository is not public: {repo_id}")
        for name in actual:
            downloaded = hf_hub_download(repo_id, name, revision=commit.oid,
                                          endpoint="https://huggingface.co")
            if sha256(downloaded) != sha256(folder / name):
                raise AssertionError(f"Remote checksum mismatch: {repo_id}/{name}")
        published.append({"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                          "revision": commit.oid, "verified_files": sorted(actual)})
        args.record.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFIED https://huggingface.co/{repo_id} @ {commit.oid}", flush=True)


if __name__ == "__main__":
    main()

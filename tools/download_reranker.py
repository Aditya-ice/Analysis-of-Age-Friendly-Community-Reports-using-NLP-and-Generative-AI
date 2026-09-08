"""Download checksummed official model artifacts at build/admin time, never on a request."""

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "contracts/reranker-manifest.json").read_text())
    destination = root / "models/reranker"
    destination.mkdir(parents=True, exist_ok=True)
    model = (
        "onnx/model_qint8_arm64.onnx"
        if platform.machine().lower() in ("arm64", "aarch64")
        else "onnx/model_quint8_avx2.onnx"
    )
    for name in ("tokenizer.json", model):
        record = manifest["files"][name]
        final = destination / Path(name).name
        if final.exists() and hashlib.sha256(final.read_bytes()).hexdigest() == record["sha256"]:
            continue
        url = f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{name}"
        with tempfile.NamedTemporaryFile(dir=destination, suffix=".part") as output:
            digest, size = hashlib.sha256(), 0
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    for block in response.iter_bytes():
                        size += len(block)
                        if size > record["bytes"]:
                            raise ValueError("Artifact exceeds pinned size")
                        digest.update(block)
                        output.write(block)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("Reranker artifact checksum mismatch")
            output.flush()
            # Copy from the verified temporary file; promote through an atomic rename.
            temporary = final.with_suffix(".verified")
            temporary.write_bytes(Path(output.name).read_bytes())
            os.replace(temporary, final)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()

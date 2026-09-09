"""Quantized CPU cross encoder. No PyTorch, downloads, or remote inference at runtime."""

import asyncio
import hashlib
import json
import math
import platform
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxRanker:
    def __init__(self, directory: Path, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text())
        name = (
            "onnx/model_qint8_arm64.onnx"
            if platform.machine().lower() in ("arm64", "aarch64")
            else "onnx/model_quint8_avx2.onnx"
        )
        for filename in ("tokenizer.json", name):
            path = directory / Path(filename).name
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != manifest["files"][filename]["sha256"]:
                raise ValueError("Reranker artifact checksum mismatch")
        self.version = manifest["revision"]
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(directory / Path(name).name),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.lock = asyncio.Lock()
        self.ready = True

    def score(self, question: str, content: str, *, stop: threading.Event | None = None):
        # Full query up to 128 WordPiece tokens. Longer inputs fail explicitly into RRF.
        query = self.tokenizer.encode(question, add_special_tokens=False).ids
        if len(query) > 128:
            raise ValueError("Question exceeds reranking query budget")
        passage = self.tokenizer.encode(content, add_special_tokens=False).ids
        window, overlap = 512 - len(query) - 3, 64
        starts = list(range(0, max(1, len(passage)), window - overlap))
        if len(starts) > 8:
            raise ValueError("Candidate exceeds bounded reranking windows")
        sequences = []
        for start in starts:
            ids = [101] + query + [102] + passage[start : start + window] + [102]
            types = [0] * (len(query) + 2) + [1] * (len(ids) - len(query) - 2)
            sequences.append((ids, types))
            if start + window >= len(passage):
                break
        best = -float("inf")
        names = {item.name for item in self.session.get_inputs()}
        for offset in range(0, len(sequences), 2):
            if stop is not None and stop.is_set():
                raise RuntimeError("Reranking cancelled")
            batch = sequences[offset : offset + 2]
            length = max(len(ids) for ids, _ in batch)
            feeds = {
                "input_ids": np.array(
                    [ids + [0] * (length - len(ids)) for ids, _ in batch], dtype=np.int64
                ),
                "attention_mask": np.array(
                    [[1] * len(ids) + [0] * (length - len(ids)) for ids, _ in batch], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [t + [0] * (length - len(t)) for _, t in batch], dtype=np.int64
                ),
            }
            scores = self.session.run(None, {k: v for k, v in feeds.items() if k in names})[0]
            best = max(best, float(scores.max()))
        if not math.isfinite(best):
            raise ValueError("Invalid reranker score")
        return 1 / (1 + math.exp(-max(-60, min(60, best))))

    def _rank(self, query, candidates, stop):
        for candidate in candidates:
            if stop.is_set():
                raise RuntimeError("Reranking cancelled")
            candidate.score = self.score(
                query, candidate.title + "\n" + candidate.content, stop=stop
            )
        return sorted(candidates, key=lambda c: (-c.score, str(c.id)))

    async def rerank(self, query, candidates):
        async with self.lock:
            stop = threading.Event()
            task = asyncio.create_task(asyncio.to_thread(self._rank, query, candidates, stop))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Finish only the current native batch, not every remaining candidate.
                # Retain the lock until the native worker has actually exited.
                stop.set()
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue  # Repeated disconnect/timeout must not release a live worker.
                    except Exception:
                        break
                try:
                    task.result()
                except Exception:
                    pass
                raise

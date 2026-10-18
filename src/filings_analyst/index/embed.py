"""Dense embedding providers.

* ``lsa``: TF-IDF + truncated SVD fitted on your own corpus. No downloads, no
  API, deterministic. A weak but honest dense baseline.
* ``fastembed``: local ONNX sentence embeddings (e.g. BAAI/bge-small-en-v1.5).
  Downloads the model from Hugging Face on first use.
* ``openai``: any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, Azure,
  vLLM, Ollama, Voyage-compatible gateways).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def save(self, directory: Path) -> None: ...


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 384, seed: int = 0):
        self.dims, self.seed = dims, seed
        self._vec = None
        self._components: np.ndarray | None = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Prune rare and ubiquitous terms on real corpora; keep everything on tiny ones.
        small = len(texts) < 200
        self._vec = TfidfVectorizer(sublinear_tf=True, min_df=1 if small else 2, max_df=1.0 if small else 0.5,
                                    ngram_range=(1, 2), max_features=300_000, dtype=np.float32)
        tfidf = self._vec.fit_transform(texts)
        dims = min(self.dims, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=max(dims, 2), random_state=self.seed)
        svd.fit(tfidf)
        self._components = svd.components_.astype(np.float32)

    def _project(self, texts: list[str]) -> np.ndarray:
        if self._vec is None or self._components is None:
            raise RuntimeError("LSA embedder is not fitted")
        return _normalize(np.asarray(self._vec.transform(texts) @ self._components.T))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._project(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._project([text])[0]

    def save(self, directory: Path) -> None:
        # Plain arrays + JSON rather than pickle, so loading an index never executes code.
        directory.mkdir(parents=True, exist_ok=True)
        vocab = {k: int(v) for k, v in self._vec.vocabulary_.items()}
        (directory / "lsa_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        np.save(directory / "lsa_idf.npy", self._vec.idf_.astype(np.float32))
        np.save(directory / "lsa_components.npy", self._components)

    @classmethod
    def load(cls, directory: Path) -> LSAEmbedder:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vocab = json.loads((directory / "lsa_vocab.json").read_text(encoding="utf-8"))
        obj = cls()
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), vocabulary=vocab, dtype=np.float32)
        vec.idf_ = np.load(directory / "lsa_idf.npy")
        obj._vec = vec
        obj._components = np.load(directory / "lsa_components.npy")
        obj.dims = obj._components.shape[0]
        return obj


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        self.model_name, self.batch_size = model, batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError("pip install 'filings-analyst[fastembed]' to use this embedder") from exc
            self._model = TextEmbedding(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return _normalize(np.array(list(model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        return _normalize(np.array(list(model.query_embed([text]))))[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fastembed.json").write_text(json.dumps({"model": self.model_name}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FastEmbedEmbedder:
        return cls(json.loads((directory / "fastembed.json").read_text(encoding="utf-8"))["model"])


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", batch_size: int = 128, client: httpx.Client | None = None):
        self.model, self.base_url, self.api_key_env, self.batch_size = model, base_url.rstrip("/"), api_key_env, batch_size
        self._client = client or httpx.Client(timeout=60)

    def fit(self, texts: list[str]) -> None:
        return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        headers = {"Authorization": f"Bearer {os.environ.get(self.api_key_env, '')}"}
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.post(f"{self.base_url}/embeddings", headers=headers,
                                     json={"model": self.model, "input": texts[i:i + self.batch_size]})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return _normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "openai.json").write_text(
            json.dumps({"model": self.model, "base_url": self.base_url, "api_key_env": self.api_key_env}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> OpenAIEmbedder:
        cfg = json.loads((directory / "openai.json").read_text(encoding="utf-8"))
        return cls(cfg["model"], cfg["base_url"], cfg["api_key_env"])


PROVIDERS = {"lsa": LSAEmbedder, "fastembed": FastEmbedEmbedder, "openai": OpenAIEmbedder}


def make_embedder(name: str, **kwargs) -> Embedder:
    if name not in PROVIDERS:
        raise ValueError(f"unknown embedder '{name}' (choose from {', '.join(PROVIDERS)})")
    return PROVIDERS[name](**kwargs)


def load_embedder(name: str, directory: Path) -> Embedder:
    return PROVIDERS[name].load(directory)

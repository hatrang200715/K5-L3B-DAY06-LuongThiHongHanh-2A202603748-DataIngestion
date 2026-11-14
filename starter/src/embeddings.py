"""Provider adapters for real text embedding APIs."""
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Call OpenAI, OpenRouter, or Gemini embedding APIs."""

    _OPENAI_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    _GEMINI_DIMENSIONS = {
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }
    _API_URLS = {
        "openai": "https://api.openai.com/v1/embeddings",
        "openrouter": "https://openrouter.ai/api/v1/embeddings",
    }
    _API_KEYS = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    def __init__(self, model_name: str) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        if ":" in model_name:
            provider, model = model_name.split(":", maxsplit=1)
            provider = provider.lower()
        else:
            configured_providers = [
                candidate
                for candidate, key in self._API_KEYS.items()
                if os.getenv(key)
            ]
            if len(configured_providers) != 1:
                raise ValueError(
                    "A bare model name requires exactly one configured provider API key; "
                    "otherwise use 'openai:<model>', 'openrouter:<model>', or "
                    "'gemini:<model>'"
                )
            provider = configured_providers[0]
            model = model_name.strip()
            if provider == "openrouter" and model in self._OPENAI_DIMENSIONS:
                model = f"openai/{model}"

        model = model.strip()
        if provider not in self._API_KEYS:
            raise ValueError(
                f"Unsupported embedding provider {provider!r}; "
                "choose openai, openrouter, or gemini"
            )
        if not model:
            raise ValueError("model name after the provider prefix cannot be empty")

        key_name = self._API_KEYS[provider]
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(f"{key_name} is required for the {provider} provider")

        configured_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        known_dimensions = self._known_dimensions(provider, model)
        if configured_dimensions:
            try:
                dimensions = int(configured_dimensions)
            except ValueError as exc:
                raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer") from exc
        elif known_dimensions is not None:
            dimensions = known_dimensions
        else:
            raise ValueError(
                f"Unknown dimensions for {provider} model {model!r}; "
                "set EMBEDDING_DIMENSIONS to the model's output size"
            )
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer")

        self.provider = provider
        self.model = model
        self.model_name = f"{provider}:{model}"
        self.dimensions = dimensions
        self._api_key = api_key

    @classmethod
    def _known_dimensions(cls, provider: str, model: str) -> int | None:
        model_id = model.rsplit("/", maxsplit=1)[-1]
        if provider in ("openai", "openrouter"):
            return cls._OPENAI_DIMENSIONS.get(model_id)
        return cls._GEMINI_DIMENSIONS.get(model_id)

    @staticmethod
    def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach embedding API: {exc.reason}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Embedding API returned invalid JSON") from exc

    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.model.rsplit("/", maxsplit=1)[-1].startswith("text-embedding-3-"):
            payload["dimensions"] = self.dimensions

        response = self._post_json(self._API_URLS[self.provider], headers, payload)
        try:
            rows = response["data"]
            if len(rows) != len(texts):
                raise RuntimeError(
                    f"Embedding API returned {len(rows)} vectors for {len(texts)} inputs"
                )
            vectors_by_index = {
                row["index"]: self._validate_vector(row["embedding"])
                for row in rows
            }
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Embedding API response is missing vector data") from exc
        if set(vectors_by_index) != set(range(len(texts))):
            raise RuntimeError("Embedding API returned an incomplete or duplicate batch")
        return [vectors_by_index[index] for index in range(len(texts))]

    def _embed_gemini(self, texts: list[str], task_type: str) -> list[list[float]]:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:embedContent"
        )
        headers = {"x-goog-api-key": self._api_key}
        vectors = []
        for text in texts:
            response = self._post_json(
                endpoint,
                headers,
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimensions,
                },
            )
            try:
                vectors.append(self._validate_vector(response["embedding"]["values"]))
            except (KeyError, TypeError) as exc:
                raise RuntimeError("Gemini response is missing embedding values") from exc
        return vectors

    def _validate_vector(self, values: Any) -> list[float]:
        if not isinstance(values, list) or len(values) != self.dimensions:
            actual = len(values) if isinstance(values, list) else "unknown"
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {actual}"
            )
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Embedding API returned non-numeric vector values") from exc
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Embedding API returned non-finite vector values")
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts in order using the configured provider."""
        if not texts:
            return []
        if self.provider == "gemini":
            return self._embed_gemini(texts, "RETRIEVAL_DOCUMENT")
        return self._embed_openai_compatible(texts)

    def embed_query(self, question: str) -> list[float]:
        """Embed a query using a task compatible with document embeddings."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if self.provider == "gemini":
            return self._embed_gemini([question], "RETRIEVAL_QUERY")[0]
        return self._embed_openai_compatible([question])[0]

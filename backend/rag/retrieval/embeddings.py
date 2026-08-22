from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from ..config import PROJECT_ROOT


def embed_texts(
    texts: Sequence[str],
    model: str,
    client: OpenAI | None = None,
    dimensions: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    load_dotenv(PROJECT_ROOT / ".env")
    openai_client = client or OpenAI()
    request: dict[str, object] = {"model": model, "input": list(texts)}
    if dimensions is not None:
        request["dimensions"] = dimensions
    response = openai_client.embeddings.create(**request)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]

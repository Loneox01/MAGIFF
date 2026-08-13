from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from .config import PROJECT_ROOT


def embed_texts(
    texts: Sequence[str],
    model: str,
    client: OpenAI | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    load_dotenv(PROJECT_ROOT / ".env")
    openai_client = client or OpenAI()
    response = openai_client.embeddings.create(model=model, input=list(texts))
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]

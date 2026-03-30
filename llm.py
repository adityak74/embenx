from typing import List, Optional

import litellm


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32, truncate_dim: Optional[int] = None):
        """
        model_name should be in LiteLLM format, e.g., 'ollama/nomic-embed-text'.
        batch_size is the number of texts to embed in a single request.
        truncate_dim: If set, truncates the embeddings to this dimension (Matryoshka).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.total_tokens_approx = 0

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using LiteLLM with batching.
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = litellm.embedding(model=self.model_name, input=batch)

                for data in response["data"]:
                    emb = data["embedding"]
                    if self.truncate_dim:
                        emb = emb[: self.truncate_dim]
                    all_embeddings.append(emb)

                if "usage" in response:
                    self.total_tokens_approx += response["usage"].get("total_tokens", 0)
                else:
                    # Fallback token estimation
                    for text in batch:
                        self.total_tokens_approx += len(text.split())

            except Exception as e:
                print(
                    f"Error generating embeddings for batch {i//self.batch_size} with {self.model_name}: {e}"
                )
                # We could return partial embeddings but it might break the benchmark
                return []

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        embs = self.embed_texts([query])
        return embs[0] if embs else []

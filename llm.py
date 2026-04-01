from typing import List, Optional
import base64
import os
from io import BytesIO
import litellm

try:
    from PIL import Image
except ImportError:
    Image = None

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
                # Basic check for image paths to trigger multimodal path
                if all(os.path.exists(t) and (t.endswith('.jpg') or t.endswith('.png')) for t in batch):
                    return self._embed_images(batch)

                response = litellm.embedding(model=self.model_name, input=batch)

                for data in response["data"]:
                    emb = data["embedding"]
                    if self.truncate_dim:
                        emb = emb[: self.truncate_dim]
                    all_embeddings.append(emb)

                if "usage" in response:
                    self.total_tokens_approx += response["usage"].get("total_tokens", 0)
                else:
                    for text in batch:
                        self.total_tokens_approx += len(text.split())

            except Exception as e:
                print(f"Error generating embeddings for {self.model_name}: {e}")
                return []

        return all_embeddings

    def _embed_images(self, paths: List[str]) -> List[List[float]]:
        """
        Multimodal embedding for images.
        """
        all_embeddings = []
        for path in paths:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{encoded_string}"
                
                # Some models support 'input' as list of image urls
                response = litellm.embedding(
                    model=self.model_name,
                    input=[{"type": "image_url", "image_url": {"url": data_url}}]
                )
                all_embeddings.append(response["data"][0]["embedding"])
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        embs = self.embed_texts([query])
        return embs[0] if embs else []

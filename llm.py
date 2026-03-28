import litellm
from typing import List

class Embedder:
    def __init__(self, model_name: str):
        """
        model_name should be in LiteLLM format, e.g., 'ollama/nomic-embed-text'
        or 'openai/text-embedding-3-small'.
        """
        self.model_name = model_name
        self.total_tokens_approx = 0

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using LiteLLM.
        """
        embeddings = []
        try:
            # LiteLLM embedding call
            response = litellm.embedding(
                model=self.model_name,
                input=texts
            )
            
            # Extract embeddings and track token usage
            for data in response["data"]:
                embeddings.append(data["embedding"])
            
            if "usage" in response:
                self.total_tokens_approx += response["usage"].get("total_tokens", 0)
            else:
                # Fallback if usage is not provided
                for text in texts:
                    self.total_tokens_approx += len(text.split())
                    
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings with {self.model_name}: {e}")
            return []
        
    def embed_query(self, query: str) -> List[float]:
        embs = self.embed_texts([query])
        return embs[0] if embs else []

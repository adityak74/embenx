import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from typing import List, Dict, Any, Tuple
import os
from .base import BaseIndexer

class WeaviateIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("Weaviate", dimension)
        # Using embedded Weaviate for local benchmarking
        self.client = weaviate.connect_to_embedded(
            version="1.27.0",
            persistence_data_path="./weaviate_data",
            port=8099,
            grpc_port=50060,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            )
        )
        self.collection_name = "Benchmark"
        
        # Cleanup if exists
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
            
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="metadata_json", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            )
        )
        self.collection = self.client.collections.get(self.collection_name)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        import json
        with self.collection.batch.dynamic() as batch:
            for emb, meta in zip(embeddings, metadata):
                batch.add_object(
                    properties={"metadata_json": json.dumps(meta)},
                    vector=emb
                )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        import json
        response = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["distance"]
        )
        
        results = []
        for obj in response.objects:
            meta = json.loads(obj.properties["metadata_json"])
            dist = obj.metadata.distance if obj.metadata.distance is not None else 0.0
            results.append((meta, float(dist)))
        return results

    def get_size(self) -> int:
        # Estimation based on data folder if exists
        size = 0
        try:
            if os.path.exists("./weaviate_data"):
                for root, dirs, files in os.walk("./weaviate_data"):
                    for f in files:
                        size += os.path.getsize(os.path.join(root, f))
        except Exception:
            pass
        return size

    def cleanup(self) -> None:
        self.client.close()
        import shutil
        try:
            if os.path.exists("./weaviate_data"):
                shutil.rmtree("./weaviate_data")
        except Exception:
            pass

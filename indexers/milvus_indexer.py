from pymilvus import MilvusClient, DataType
import tempfile
from typing import List, Dict, Any, Tuple
import os
from .base import BaseIndexer

class MilvusIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("Milvus", dimension)
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.client = MilvusClient(uri=self.temp_file.name)
        self.collection_name = "benchmark"
        
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type="COSINE"
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        data = []
        for emb, meta in zip(embeddings, metadata):
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            
            row = {"vector": emb, **clean_meta}
            data.append(row)
            
        if data:
            self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["*"]
        )
        
        out = []
        if res and len(res[0]) > 0:
            for hit in res[0]:
                entity = hit['entity']
                dist = hit['distance']
                entity.pop('id', None)
                entity.pop('vector', None)
                out.append((entity, float(dist)))
        return out

    def get_size(self) -> int:
        if os.path.exists(self.temp_file.name):
            return os.path.getsize(self.temp_file.name)
        return 0

    def cleanup(self) -> None:
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

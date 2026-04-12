import numpy as np

from embenx import Collection


def run_usearch_example():
    print("--- USearch Quantization Example ---")
    dim = 512
    n = 1000
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i} for i in range(n)]

    # 1. Standard USearch (Float32)
    print("\n1. USearch-F32")
    col_f32 = Collection(dimension=dim, indexer_type="usearch")
    col_f32.add(vectors, metadata)
    print(f" Size: {col_f32.indexer.get_size() / 1024:.2f} KB")

    # 2. USearch-F16 (Half precision)
    print("\n2. USearch-F16")
    col_f16 = Collection(dimension=dim, indexer_type="usearch-f16")
    col_f16.add(vectors, metadata)
    print(f" Size: {col_f16.indexer.get_size() / 1024:.2f} KB")

    # 3. USearch-I8 (Int8 quantization)
    print("\n3. USearch-I8")
    col_i8 = Collection(dimension=dim, indexer_type="usearch-i8")
    col_i8.add(vectors, metadata)
    print(f" Size: {col_i8.indexer.get_size() / 1024:.2f} KB")


if __name__ == "__main__":
    run_usearch_example()

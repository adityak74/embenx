import os
import shutil

import numpy as np
from embenx.core import StateCollection


def run_ssm_example():
    print("--- State Space Model (SSM) Hydration Example ---")
    
    # 1. Setup a StateCollection for Mamba-style hidden states
    dim = 128
    state_dim = 2048 # hidden state size
    col = StateCollection(name="mamba_memory", dimension=dim)
    
    # 2. Simulate data
    n = 3
    vectors = np.random.rand(n, dim).astype(np.float32)
    # hidden states 'h' [batch, state_dim]
    states = np.random.rand(n, state_dim).astype(np.float32)
    metadata = [{"id": f"s_{i}", "text": f"State at t={i}"} for i in range(n)]
    
    # 3. Add states to collection
    print("Indexing latent states and their hidden representations...")
    col.add_states(vectors, states, metadata)

    # 4. Retrieve and hydrate
    print("\nRetrieving state for a query...")
    query = np.random.rand(dim).astype(np.float32)
    results = col.search(query, top_k=1)
    
    best_meta, score = results[0]
    print(f" Found matching state: {best_meta['text']}")
    
    print("Hydrating hidden state h0...")
    h0 = col.get_state(best_meta)
    print(f" Loaded hidden state shape: {h0.shape}")

    # Cleanup
    if os.path.exists("states_mamba_memory"):
        shutil.rmtree("states_mamba_memory")

if __name__ == "__main__":
    run_ssm_example()

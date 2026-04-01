from embenx.core import Session
import numpy as np
import time
import shutil
import os

def run_session_example():
    print("--- Managed Agentic Session Example ---")
    
    # 1. Start a session for 'agent_alpha'
    # Sessions automatically persist to disk and handle temporal context
    sess = Session(session_id="agent_alpha", dimension=128)
    
    # 2. Add interactions over time
    print("Recording conversation history...")
    
    # Interaction 1 (Now)
    sess.add_interaction(np.random.rand(128), "What is the capital of France?", role="user")
    
    # Interaction 2 (Simulate gap)
    time.sleep(0.1)
    sess.add_interaction(np.random.rand(128), "Paris is the capital of France.", role="assistant")

    # 3. Retrieve context for a new query with recency bias
    print("\nRetrieving context for: 'Tell me more about that city'")
    query_vec = np.random.rand(128)
    
    # recency_weight=0.6 means 60% importance to recent messages
    context = sess.retrieve_context(query_vec, top_k=2, recency_weight=0.6)
    
    for i, (meta, score) in enumerate(context):
        print(f" Context {i+1}: {meta['text']} (Score: {score:.4f})")

    # 4. Clean up
    print("\nSession data persisted at: .embenx_sessions/agent_alpha.parquet")
    # sess.cleanup() # Uncomment to delete

if __name__ == "__main__":
    run_session_example()

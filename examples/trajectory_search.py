from embenx import Collection
import numpy as np

def run_trajectory_example():
    print("--- Trajectory Retrieval Example ---")
    
    # 1. Setup a collection for robot states
    # Imagine each vector is a [position_x, position_y, velocity, force]
    dim = 4
    col = Collection(dimension=dim, indexer_type="faiss")
    
    # 2. Add some "experienced" trajectories
    # Trajectory 1: Moving forward
    traj1 = np.array([
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [2, 0, 1, 0]
    ])
    # Trajectory 2: Turning
    traj2 = np.array([
        [0, 0, 1, 0],
        [0.5, 0.5, 1, 0.1],
        [1, 1, 0.5, 0.2]
    ])
    
    # In a real World Model, we might index the whole trajectory as a pooled vector
    # For now, we add the mean of each trajectory to the collection
    print("Indexing expert trajectories...")
    col.add([np.mean(traj1, axis=0)], [{"id": "forward_action", "type": "trajectory"}])
    col.add([np.mean(traj2, axis=0)], [{"id": "turn_action", "type": "trajectory"}])

    # 3. Search for a similar current trajectory
    # Our agent is currently doing something like turning
    current_traj = np.array([
        [0.1, 0.1, 0.9, 0],
        [0.4, 0.4, 0.8, 0.1]
    ])
    
    print("\nSearching for similar past experiences given current trajectory...")
    results = col.search_trajectory(current_traj, top_k=1, pooling="mean")
    
    for meta, dist in results:
        print(f" Found matching behavior: {meta['id']} (Distance: {dist:.4f})")

if __name__ == "__main__":
    run_trajectory_example()

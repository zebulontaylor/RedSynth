import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redsynth.routing import RoutingGrid, a_star

def test_penalty_avoidance():
    # Setup a simple grid
    # Node A at (10, 10, 10) size 5x5x5
    # Start at (5, 10, 10)
    # End at (15, 10, 10)
    # Direct path goes through Node A
    
    positions = {
        'NodeA': (20.0, 20.0, 20.0) # World coords
    }
    nodes_data = {
        'NodeA': {
            'dims': (10, 10, 10), # World dims -> 5x5x5 grid dims approx
            'pin_locations': {}
        }
    }
    
    grid = RoutingGrid(positions, nodes_data)
    
    # Get node coords
    node_coords = grid.get_node_coords('NodeA')
    print(f"Node A occupies {len(node_coords)} voxels")
    
    # Define start and end points (outside the node but on opposite sides)
    # Node center is at grid (10, 10, 10)
    # Bounds approx (8,8,8) to (12,12,12)
    
    start = (5, 10, 10)
    end = (15, 10, 10)
    
    # Ensure start/end are not blocked
    grid.blocked_coords.discard(start)
    grid.blocked_coords.discard(end)
    
    # Case 1: No penalty (should go straight through if allowed)
    allowed_points = set(node_coords)
    allowed_points.add(start)
    allowed_points.add(end)
    
    path_no_penalty = a_star(start, end, grid, allowed_points, penalty_points=None)
    
    print("\n--- No Penalty ---")
    if path_no_penalty:
        print(f"Path length: {len(path_no_penalty)}")
        through_node = any(p in node_coords for p in path_no_penalty)
        print(f"Goes through node: {through_node}")
    else:
        print("No path found")

    # Case 2: With penalty (should go around)
    path_with_penalty = a_star(start, end, grid, allowed_points, penalty_points=node_coords)
    
    print("\n--- With Penalty ---")
    if path_with_penalty:
        print(f"Path length: {len(path_with_penalty)}")
        through_node = any(p in node_coords for p in path_with_penalty)
        print(f"Goes through node: {through_node}")
        
        if not through_node:
            print("SUCCESS: Path avoided the node!")
        else:
            print("FAILURE: Path still went through the node.")
    else:
        print("No path found with penalty")

if __name__ == "__main__":
    test_penalty_avoidance()


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redsynth.routing import RoutingGrid

def test_vertical_slope():
    # Setup grid
    # Expand grid to cover (0,0,0) to (20,20,20)
    positions = {'A': (0,0,0), 'B': (20,20,20)} 
    nodes_data = {'A': {'dims': (1,1,1)}, 'B': {'dims': (1,1,1)}}
    grid = RoutingGrid(positions, nodes_data)
    
    print("--- Test 1: Vertical Slope Same Direction ---")
    # Wire 1: Slopes UP and EAST (+X, +Y)
    # From (10, 10, 10) to (11, 11, 10)
    path1 = [(10, 10, 10), (11, 11, 10)]
    grid.add_path(path1, "net1")
    print(f"Added Wire 1: {path1}")
    
    # Wire 2: Wants to slope UP and EAST (+X, +Y) directly below Wire 1
    # From (10, 9, 10) to (11, 10, 10)
    start = (10, 9, 10)
    target = (11, 10, 10)
    
    print(f"Checking move for Wire 2: {start} -> {target}")
    
    neighbors = grid.get_neighbors(start)
    found = False
    for n, cost in neighbors:
        if n == target:
            found = True
            break
            
    if found:
        print("PASS: Wire 2 allowed to slope up parallel to Wire 1.")
    else:
        print("FAIL: Wire 2 blocked from sloping up parallel to Wire 1.")

    # Clean up for next test
    grid.remove_path(path1, "net1")
    
    print("\n--- Test 2: Vertical Slope Different Direction ---")
    # Wire 1: Slopes UP and EAST (+X, +Y)
    # From (10, 10, 10) to (11, 11, 10)
    grid.add_path(path1, "net1")
    print(f"Added Wire 1: {path1}")
    
    # Wire 3: Wants to slope UP and WEST (-X, +Y) directly below Wire 1
    # From (10, 9, 10) to (9, 10, 10)
    # Note: The start point (10, 9, 10) is below Wire 1's start (10, 10, 10).
    # The move is (-1, 1, 0).
    # The check_pos will be (10, 10, 10) (since dy=1).
    # Wire 1 at (10, 10, 10) has direction (1, 1, 0).
    # Wire 3 move is (-1, 1, 0).
    # These are NOT the same. Should be blocked.
    
    start3 = (10, 9, 10)
    target3 = (9, 10, 10)
    
    print(f"Checking move for Wire 3: {start3} -> {target3}")
    
    neighbors = grid.get_neighbors(start3)
    found = False
    for n, cost in neighbors:
        if n == target3:
            found = True
            break
            
    if not found:
        print("PASS: Wire 3 blocked from sloping up in different direction.")
    else:
        print("FAIL: Wire 3 allowed to slope up in different direction.")

    # Clean up
    grid.remove_path(path1, "net1")

    print("\n--- Test 3: Vertical Slope Stacked Downward ---")
    # Wire 4: Slopes DOWN and EAST (+X, -Y)
    # From (10, 10, 10) to (11, 9, 10)
    path4 = [(10, 10, 10), (11, 9, 10)]
    grid.add_path(path4, "net4")
    print(f"Added Wire 4: {path4}")
    
    # Wire 5: Wants to slope DOWN and EAST (+X, -Y) directly ABOVE Wire 4
    # From (10, 11, 10) to (11, 10, 10)
    start5 = (10, 11, 10)
    target5 = (11, 10, 10)
    
    # For downward move (dy=-1), check_pos is y-1 => (10, 10, 10).
    # Wire 4 at (10, 10, 10) has direction (1, -1, 0).
    # Wire 5 move is (1, -1, 0).
    # Should match.
    
    print(f"Checking move for Wire 5: {start5} -> {target5}")
    
    neighbors = grid.get_neighbors(start5)
    found = False
    for n, cost in neighbors:
        if n == target5:
            found = True
            break
            
    if found:
        print("PASS: Wire 5 allowed to slope down parallel to Wire 4.")
    else:
        print("FAIL: Wire 5 blocked from sloping down parallel to Wire 4.")


if __name__ == "__main__":
    test_vertical_slope()

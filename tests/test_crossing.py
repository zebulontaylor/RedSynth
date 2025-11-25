
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redsynth.routing import RoutingGrid

def test_crossing():
    # Setup grid
    # Expand grid to cover (0,0,0) to (100,100,100)
    positions = {'A': (0,0,0), 'B': (100,100,100)} 
    nodes_data = {'A': {'dims': (1,1,1)}, 'B': {'dims': (1,1,1)}}
    grid = RoutingGrid(positions, nodes_data)
    
    # Add a wire along X-axis at y=10, z=10
    # From (8, 10, 10) to (12, 10, 10)
    path1 = [(8, 10, 10), (9, 10, 10), (10, 10, 10), (11, 10, 10), (12, 10, 10)]
    grid.add_path(path1, "net1")
    
    print("Wire 1 added along X-axis at y=10, z=10.")
    
    # Test 1: Cross along Z-axis at (10, 10, 10)
    # Move from (10, 10, 9) to (10, 10, 10)
    start = (10, 10, 9)
    neighbors = grid.get_neighbors(start)
    found = False
    for n, cost in neighbors:
        if n == (10, 10, 10):
            found = True
            break
    
    if found:
        print("PASS: Crossing along Z-axis allowed.")
    else:
        print("FAIL: Crossing along Z-axis blocked.")

    # Test 2: Overlap along X-axis at (10, 10, 10)
    # Move from (9, 10, 10) to (10, 10, 10)
    start = (9, 10, 10)
    neighbors = grid.get_neighbors(start, allowed_points={start})
    found = False
    for n, cost in neighbors:
        if n == (10, 10, 10):
            found = True
            break
            
    if not found:
        print("PASS: Overlap along X-axis blocked.")
    else:
        print("FAIL: Overlap along X-axis allowed.")

    # Test 3: Cross along Y-axis at (10, 10, 10)
    # Move from (10, 9, 10) to (10, 10, 10)
    start = (10, 9, 10)
    neighbors = grid.get_neighbors(start)
    found = False
    for n, cost in neighbors:
        if n == (10, 10, 10):
            found = True
            break
            
    if not found:
        print("PASS: Crossing along Y-axis blocked.")
    else:
        print("FAIL: Crossing along Y-axis allowed.")

    # Test 4: Corner Case
    # Add wire 2: (20, 10, 10) -> (21, 10, 10) -> (21, 10, 11)
    # Corner at (21, 10, 10).
    path2 = [(20, 10, 10), (21, 10, 10), (21, 10, 11)]
    grid.add_path(path2, "net2")
    
    # At (21, 10, 10), directions are (1,0,0) and (0,0,1).
    # Try to cross along Y-axis (0, 1, 0).
    # Perpendicular to (1,0,0)? Yes.
    # Perpendicular to (0,0,1)? Yes.
    # Should allow.
    
    start = (21, 9, 10)
    neighbors = grid.get_neighbors(start)
    found = False
    for n, cost in neighbors:
        if n == (21, 10, 10):
            found = True
            break
            
    if not found:
        print("PASS: Crossing corner along Y-axis blocked.")
    else:
        print("FAIL: Crossing corner along Y-axis allowed.")
        
    # Try to cross corner along X-axis (1, 0, 0).
    # Perpendicular to (1,0,0)? No.
    # Should block.
    start = (22, 10, 10)
    neighbors = grid.get_neighbors(start)
    found = False
    for n, cost in neighbors:
        if n == (21, 10, 10):
            found = True
            break
            
    if not found:
        print("PASS: Crossing corner along X-axis blocked.")
    else:
        print("FAIL: Crossing corner along X-axis allowed.")

if __name__ == "__main__":
    test_crossing()

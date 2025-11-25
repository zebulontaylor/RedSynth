from typing import Dict, List, Tuple

def handle_straight(start: Tuple[int, int, int], end: Tuple[int, int, int], grid: Dict[Tuple[int, int, int], str]):
    """Handles straight moves (2 blocks long)."""
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Midpoint
    mx = x1 + dx // 2
    my = y1 + dy // 2
    mz = z1 + dz // 2
    
    grid[start] = "redstone_wire"
    grid[(mx, my, mz)] = "redstone_wire"
    
    # Support
    grid[(x1, y1 - 1, z1)] = "stone"
    grid[(mx, my - 1, mz)] = "stone"

def handle_diagonal(start: Tuple[int, int, int], end: Tuple[int, int, int], grid: Dict[Tuple[int, int, int], str]):
    """Handles diagonal moves (staircases)."""
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # We expect abs(dy) == 2 for vertical diagonals
    # And one of dx or dz to be +/- 2
    
    grid[start] = "redstone_wire"
    
    # Support for start
    grid[(x1, y1 - 1, z1)] = "stone"
    
    # Midpoint logic for staircase
    # If going UP (dy > 0):
    #   Start at (x, y, z)
    #   Step 1: (x + dx/2, y + 1, z + dz/2) -> This is the stair block
    #   End at (x + dx, y + 2, z + dz)
    
    # If going DOWN (dy < 0):
    #   Start at (x, y, z)
    #   Step 1: (x + dx/2, y - 1, z + dz/2) -> This is the stair block
    #   End at (x + dx, y - 2, z + dz)
    
    mx = x1 + dx // 2
    my = y1 + dy // 2
    mz = z1 + dz // 2
    
    grid[(mx, my, mz)] = "redstone_wire"
    grid[(mx, my - 1, mz)] = "stone"

def generate_redstone_grid(routed_paths: Dict[str, List[List[Tuple[int, int, int]]]]) -> Dict[Tuple[int, int, int], str]:
    """
    Generates a sparse grid of blocks based on routed paths.
    
    Args:
        routed_paths: Dictionary mapping net names to lists of paths (lists of coordinates).
        
    Returns:
        Dictionary mapping (x, y, z) coordinates to block type strings.
    """
    grid = {}
    
    for net_name, paths in routed_paths.items():
        for path in paths:
            if not path: continue
            
            # Add the very last point explicitly, as handlers only do start+mid
            last_p = path[-1]
            grid[last_p] = "redstone_wire"
            grid[(last_p[0], last_p[1]-1, last_p[2])] = "stone"
            
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i+1]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dz = p2[2] - p1[2]
                
                # Determine move type
                # Straight: Only one component is +/- 2, others 0
                # Diagonal: dy is +/- 2, and one of dx/dz is +/- 2
                
                is_straight = (abs(dx) == 2 and dy == 0 and dz == 0) or \
                              (abs(dy) == 2 and dx == 0 and dz == 0) or \
                              (abs(dz) == 2 and dx == 0 and dy == 0)
                              
                if is_straight:
                    handle_straight(p1, p2, grid)
                else:
                    # Assume diagonal
                    handle_diagonal(p1, p2, grid)
                    
    return grid

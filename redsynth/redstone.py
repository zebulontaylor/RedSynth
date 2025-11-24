from typing import Dict, List, Tuple

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
            prev_pos = path[0]
            for x, y, z in path:
                dx = x - prev_pos[0]
                dy = y - prev_pos[1]
                dz = z - prev_pos[2]
                
                grid[(x, y, z)] = "redstone_wire"
                support_pos = [(x, y - 1, z)]

                if abs(dx) == 2 or abs(dz) == 2:
                    mid_y = y - dy // 2
                    grid[(x - dx / 2, mid_y, z - dz / 2)] = "redstone_wire"
                    support_pos.append((x - dx / 2, mid_y - 1, z - dz / 2))

                if abs(dx) == 3 or abs(dz) == 3:
                    grid[(x - dx / 3, y, z - dz / 3)] = "redstone_wire"
                    grid[(x - 2*dx / 3, y, z - 2*dz / 3)] = "redstone_wire"
                    support_pos.append((x - dx / 3, y - 1, z - dz / 3))
                    support_pos.append((x - 2*dx / 3, y - 1, z - 2*dz / 3))
                
                for pos in support_pos:
                    if pos not in grid:
                        grid[pos] = "stone"
                
                prev_pos = (x, y, z)
                    
    return grid

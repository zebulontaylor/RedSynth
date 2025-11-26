from typing import Dict, List, Tuple, Set

def get_segment_direction(p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> str:
    """
    Determine the primary horizontal direction of a segment.
    Returns 'x' for X-axis movement, 'z' for Z-axis movement.
    """
    dx = abs(p2[0] - p1[0])
    dz = abs(p2[2] - p1[2])
    
    if dx > dz:
        return 'x'
    elif dz > dx:
        return 'z'
    else:
        # Equal or both zero (pure vertical) - default to 'x'
        return 'x'

def get_wire_positions_for_segment(
    p1: Tuple[int, int, int], 
    p2: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """
    Get all wire positions for a segment.
    
    Routing moves:
    - Cardinal: 1 unit in X or Z (flat move) - just place at start
    - Slope: 2 in X/Z + 1 in Y (staircase) - place start and midpoint
    
    Returns positions for wire placement (excludes endpoint, handled by next segment).
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    positions = []
    
    # Cardinal move: 1 unit in one horizontal axis, no vertical change
    is_cardinal = (abs(dx) == 1 and dy == 0 and dz == 0) or \
                  (abs(dz) == 1 and dx == 0 and dy == 0)
    
    if is_cardinal:
        # Just place at start point (endpoint handled by next segment or final point)
        positions.append((x1, y1, z1))
    else:
        # Slope move: 2 horizontal + 1 vertical - creates a staircase
        # Place at start and midpoint
        positions.append((x1, y1, z1))
        
        # Midpoints
        m1x = x1 + dx // 4
        m1y = y1 + dy // 4
        m1z = z1 + dz // 4
        m2x = x1 + dx // 2
        m2y = y1 + dy // 2
        m2z = z1 + dz // 2
        m3x = x1 + 3 * dx // 4
        m3y = y1 + 3 * dy // 4
        m3z = z1 + 3 * dz // 4
        positions.append((m1x, m1y, m1z))
        positions.append((m2x, m2y, m2z))
        positions.append((m3x, m3y, m3z))
    
    return positions

def generate_redstone_grid(routed_paths: Dict[str, List[List[Tuple[int, int, int]]]]) -> Dict[Tuple[int, int, int], str]:
    """
    Generates a sparse grid of blocks based on routed paths using a two-pass system
    with directional layer assignment.
    
    Two-Pass System:
    - Pass 1: Collect all redstone wire positions with directions
    - Pass 2: Place supports (glass where wire descends, stone elsewhere)
    
    Args:
        routed_paths: Dictionary mapping net names to lists of paths.
        
    Returns:
        Dictionary mapping (x, y, z) coordinates to block type strings.
    """
    # Track wire positions and their directions
    # (x, y, z) -> Dict[net_name, direction ('x' or 'z')]
    wire_positions = {}
    
    # ===================
    # PASS 1: Collect wire positions with directions
    # ===================
    
    for net_name, paths in routed_paths.items():
        for path in paths:
            if not path:
                continue
            
            # Process each segment
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                
                # Get direction for this segment
                direction = get_segment_direction(p1, p2)

                # Get wire positions for this segment
                segment_wires = get_wire_positions_for_segment(p1, p2)
                
                for pos in segment_wires:
                    if pos not in wire_positions:
                        wire_positions[pos] = {}
                    wire_positions[pos][net_name] = direction
            
            # Add the final endpoint
            if path:
                last_p = path[-1]
                final_pos = (last_p[0], last_p[1], last_p[2])
                if final_pos not in wire_positions:
                    wire_positions[final_pos] = {}
                # Use direction from last segment if available
                if len(path) >= 2:
                    direction = get_segment_direction(path[-2], path[-1])
                    wire_positions[final_pos][net_name] = direction
                else:
                    wire_positions[final_pos][net_name] = 'x'  # default
    
    # ===================
    # PASS 2: Place wires and supports
    # ===================
    
    grid = {}
    
    # First, place all redstone wires
    for pos, net_dict in wire_positions.items():
        if len(net_dict) == 1:
            grid[pos] = "redstone_wire"
        else:
            grid[pos] = "repeater"
    
    new_wire_positions = wire_positions.copy()

    # Next, place 2nd repeater for intersections - need to pick axis w/o overlap underneath
    for pos, net_dict in wire_positions.items():
        if len(net_dict) == 2:
            # Define axes: X-axis positions and Z-axis positions
            x_axis = [(pos[0] + 1, pos[1] - 2, pos[2]), (pos[0] - 1, pos[1] - 2, pos[2])]
            z_axis = [(pos[0], pos[1] - 2, pos[2] + 1), (pos[0], pos[1] - 2, pos[2] - 1)]
            axes = [x_axis, z_axis]
            axis_names = ['x', 'z']
            
            valid_axis_found = False
            above_other_intersection = len(wire_positions.get((pos[0], pos[1]-2, pos[2]), {}).keys()) > 1

            for axis, axis_name in zip(axes, axis_names):
                if (axis[0] not in wire_positions and axis[1] not in wire_positions) or above_other_intersection:
                    pos_2_minus_one = (axis[0][0], axis[0][1]+1, axis[0][2])
                    pos_2 = (axis[0][0], axis[0][1]+2, axis[0][2])
                    pos_3_minus_one = (axis[1][0], axis[1][1]+1, axis[1][2])
                    pos_3 = (axis[1][0], axis[1][1]+2, axis[1][2])

                    # Determine which net goes to which position based on wire direction
                    net_on_axis = None
                    for net_name, direction in net_dict.items():
                        if direction == axis_name:
                            net_on_axis = net_name
                            break
                    
                    if net_on_axis is not None:
                        grid[pos_2_minus_one] = "redstone_wire"
                        grid[pos_3_minus_one] = "repeater"
                    else:
                        grid[pos_2_minus_one] = "repeater"
                        grid[pos_3_minus_one] = "redstone_wire"
                    
                    new_wire_positions[pos_2_minus_one] = new_wire_positions.pop(pos_2, {net_on_axis: axis_name})
                    new_wire_positions[pos_3_minus_one] = new_wire_positions.pop(pos_3, {net_on_axis: axis_name})
                    
                    grid.pop(pos_2, None)
                    grid.pop(pos_3, None)

                    valid_axis_found = True
                    break
            if not valid_axis_found:
                print(f"No valid axis found for net {net_dict} at position {pos}")
    
    wire_positions = new_wire_positions

    # Then, place supports
    for pos, net_dict in wire_positions.items():
        x, y, z = pos
        support_pos = (x, y - 1, z)
        
        # Skip if there's already a wire at the support position (crossing case)
        if support_pos in wire_positions:
            continue
        
        # Skip if we already placed something at support position
        if support_pos in grid:
            continue
        
        adjacent_up_positions = [
            (x + 1, y + 1, z),
            (x - 1, y + 1, z),
            (x, y + 1, z + 1),
            (x, y + 1, z - 1),
        ]

        for up_pos in adjacent_up_positions:
            if up_pos in wire_positions and set(net_dict.keys()).intersection(wire_positions[up_pos].keys()):
                grid[support_pos] = "glass"
                break
        
        if support_pos not in grid:
            grid[support_pos] = "stone"
    
    return grid

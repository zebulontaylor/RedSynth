from typing import Dict, List, Tuple, Set
import collections

def get_segment_direction(p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> str:
    """
    Determine the primary horizontal direction of a segment.
    Returns '+x', '-x', '+z', or '-z' based on movement direction.
    """
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    
    if abs(dx) > abs(dz):
        return '+x' if dx > 0 else '-x'
    elif abs(dz) > abs(dx):
        return '+z' if dz > 0 else '-z'
    else:
        # Equal magnitudes or both zero - check which is non-zero
        if dx != 0:
            return '+x' if dx > 0 else '-x'
        elif dz != 0:
            return '+z' if dz > 0 else '-z'
        else:
            # Pure vertical or same point - default to '+x'
            return '+x'

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

    vertical_offset = 1 if dx != 0 else 0
    
    # Cardinal move: 1 unit in one horizontal axis, no vertical change
    is_cardinal = (abs(dx) == 2 and dy == 0 and dz == 0) or \
                  (abs(dz) == 2 and dx == 0 and dy == 0)
    
    if is_cardinal:
        mx = x1 + dx // 2
        my = y1 + dy // 2
        mz = z1 + dz // 2
        positions.append((x1, y1+vertical_offset, z1))
        positions.append((mx, my+vertical_offset, mz))
    else:
        # Slope move: 2 horizontal + 1 vertical - creates a staircase
        # Use absolute grid alignment for slope offsets to ensure crossing slopes
        # are synchronized and don't encroach on each other
        
        is_x_slope = abs(dx) > abs(dz)
        
        def slope_offset(px, pz, is_x_slope):
            """
            Calculate y-offset for slope wire based on absolute grid position.
            - x-slopes: step up at odd x positions
            - z-slopes: step up at odd z positions, plus extra offset at same-parity
                        crossings to ensure 1-level separation from x-slopes
            """
            if is_x_slope:
                return 1 if px % 2 == 1 else 0
            else:
                base_offset = 1 if pz % 2 == 1 else 0
                # Add extra offset when x and z have same parity to avoid collision
                # with x-slopes at crossing points
                crossing_offset = 1 if px % 2 == pz % 2 else 0
                return base_offset + crossing_offset
        
        # Calculate the 4 positions along the slope
        all_points = [
            (x1, y1, z1),
            (x1 + dx // 4, y1 + dy // 4, z1 + dz // 4),
            (x1 + dx // 2, y1 + dy // 2, z1 + dz // 2),
            (x1 + 3 * dx // 4, y1 + 3 * dy // 4, z1 + 3 * dz // 4),
        ]
        
        for (px, py, pz) in all_points:
            offset = slope_offset(px, pz, is_x_slope)
            positions.append((px, py + offset, pz))
    
    return positions

def generate_redstone_grid(routed_paths: Dict[str, List[List[Tuple[int, int, int]]]]) -> Tuple[Dict[Tuple[int, int, int], str], Dict[Tuple[int, int, int], Dict[str, str]]]:
    """
    Generates a sparse grid of blocks based on routed paths using a two-pass system
    with directional layer assignment.
    
    Two-Pass System:
    - Pass 1: Collect all redstone wire positions with directions
    - Pass 2: Place supports (glass where wire descends, stone elsewhere)
    
    Args:
        routed_paths: Dictionary mapping net names to lists of paths.
        
    Returns:
        Tuple of:
        - Dictionary mapping (x, y, z) coordinates to block type strings.
        - Dictionary mapping (x, y, z) to Dict[net_name, direction] for wire positions
          (useful for visualizing multi-net positions)
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
                    wire_positions[final_pos][net_name] = '+x'  # default
    
    # ===================
    # PASS 2: Place wires and supports
    # ===================
    
    grid = {}
    new_wire_positions = wire_positions.copy()
    
    # First, place all redstone wires
    for pos, net_dict in wire_positions.items():
        pos_up = (pos[0], pos[1]+1, pos[2])
        # If a net has vertical overlap, skip this position
        if net_dict.keys() == wire_positions.get(pos_up, {}).keys():
            new_wire_positions.pop(pos, None)
            continue
        if len(net_dict) == 1:
            grid[pos] = "redstone_wire"
        else:
            grid[pos] = "repeater"
    
    wire_positions = new_wire_positions.copy()

    # Next, place repeaters for intersections
    for pos, net_dict in wire_positions.items():
        # Only do this pass along the x axis
        if len(net_dict.values()) > 1:
            #raise ValueError(f"Multiple nets at position {pos}")
            print(f"Multiple nets at position {pos}: {net_dict}")
            continue
            
        pos_down = (pos[0], pos[1]-1, pos[2])
        pos_up = (pos[0], pos[1]+1, pos[2])

        is_x = list(net_dict.values())[0] in ('+x', '-x')

        # Intersection detected
        if pos_down in wire_positions or pos_up in wire_positions:
            if is_x:
                grid[pos] = "repeater"
            else:
                grid[pos] = "stone"
                # Place the 2nd repeater in the direction of the wire
                dir_wire = list(wire_positions[pos].values())[0]
                if dir_wire[0] == '+':
                    grid[pos[0], pos[1], pos[2] + 1] = "repeater"
                else:
                    grid[pos[0], pos[1], pos[2] - 1] = "repeater"
    
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
    
    # ===================
    # PASS 3: Signal Delay Repeaters
    # ===================
    grid = apply_signal_delay_repeaters(grid, routed_paths, wire_positions)

    return grid, wire_positions

def build_net_graph(paths: List[List[Tuple[int, int, int]]]) -> Tuple[Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], Tuple[int, int, int]]:
    """
    Builds an adjacency list graph for a net from its paths.
    Uses actual wire positions (with vertical offsets applied).
    
    Returns:
        Tuple of (adjacency dict, source position)
    """
    adj = collections.defaultdict(list)
    source = None
    
    for path in paths:
        if not path:
            continue
            
        # Collect all wire positions for this entire path in order
        all_wires = []
        
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            segment_wires = get_wire_positions_for_segment(u, v)
            all_wires.extend(segment_wires)
        
        if not all_wires:
            continue
            
        # Remove consecutive duplicates while preserving order
        unique_wires = [all_wires[0]]
        for w in all_wires[1:]:
            if w != unique_wires[-1]:
                unique_wires.append(w)
        
        # Build edges between consecutive wires
        for j in range(len(unique_wires) - 1):
            p1, p2 = unique_wires[j], unique_wires[j + 1]
            if p2 not in adj[p1]:
                adj[p1].append(p2)
        
        # Set source to first wire position
        if source is None:
            source = unique_wires[0]
    
    return adj, source

def get_repeater_back_position(
    pos: Tuple[int, int, int],
    direction: str
) -> Tuple[int, int, int] | None:
    """
    Calculate the back position of a repeater based on its facing direction.
    
    Args:
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        
    Returns:
        The back position, or None if direction is unknown
    """
    x, y, z = pos
    
    # Direction indicates where signal flows TO, back is opposite
    if direction == '+x':
        return (x - 1, y, z)
    elif direction == '-x':
        return (x + 1, y, z)
    elif direction == '+z':
        return (x, y, z - 1)
    elif direction == '-z':
        return (x, y, z + 1)
    else:
        return None


def get_repeater_front_position(
    pos: Tuple[int, int, int],
    direction: str
) -> Tuple[int, int, int] | None:
    """
    Calculate the front position of a repeater based on its facing direction.
    
    Args:
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        
    Returns:
        The front position, or None if direction is unknown
    """
    x, y, z = pos
    
    # Direction indicates where signal flows TO (front)
    if direction == '+x':
        return (x + 1, y, z)
    elif direction == '-x':
        return (x - 1, y, z)
    elif direction == '+z':
        return (x, y, z + 1)
    elif direction == '-z':
        return (x, y, z - 1)
    else:
        return None


def _is_valid_repeater_connection(
    grid: Dict[Tuple[int, int, int], str],
    connection_pos: Tuple[int, int, int] | None
) -> bool:
    """
    Check if a position is a valid repeater connection point.
    
    Valid connections are:
    - A redstone_wire block directly
    - A stone block with redstone_wire on top of it
    
    Args:
        grid: The current block grid
        connection_pos: Position to check
        
    Returns:
        True if valid, False otherwise
    """
    if connection_pos is None:
        return True
    
    block = grid.get(connection_pos)
    
    # Check if position has redstone_wire directly
    if block and "redstone_wire" in block:
        return True
    
    # Check if position has stone with redstone_wire on top
    if block == "stone":
        above_pos = (connection_pos[0], connection_pos[1] + 1, connection_pos[2])
        above_block = grid.get(above_pos)
        if above_block and "redstone_wire" in above_block:
            return True

    return False


def is_valid_repeater_back(
    grid: Dict[Tuple[int, int, int], str],
    pos: Tuple[int, int, int],
    direction: str
) -> bool:
    """
    Check if a repeater at the given position would have a valid back connection.
    
    The back of a repeater must be connected to either:
    - A redstone_wire block directly
    - A stone block with redstone_wire on top of it
    
    Args:
        grid: The current block grid
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        
    Returns:
        True if valid, False otherwise
    """
    back_pos = get_repeater_back_position(pos, direction)
    return _is_valid_repeater_connection(grid, back_pos)


def is_valid_repeater_front(
    grid: Dict[Tuple[int, int, int], str],
    pos: Tuple[int, int, int],
    direction: str
) -> bool:
    """
    Check if a repeater at the given position would have a valid front connection.
    
    The front of a repeater must be connected to either:
    - A redstone_wire block directly
    - A stone block with redstone_wire on top of it
    
    Args:
        grid: The current block grid
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        
    Returns:
        True if valid, False otherwise
    """
    front_pos = get_repeater_front_position(pos, direction)
    return _is_valid_repeater_connection(grid, front_pos)


def is_valid_repeater_placement(
    grid: Dict[Tuple[int, int, int], str],
    pos: Tuple[int, int, int],
    direction: str
) -> bool:
    """
    Check if a repeater at the given position would have valid front and back connections.
    
    Args:
        grid: The current block grid
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        
    Returns:
        True if both front and back are valid, False otherwise
    """
    return is_valid_repeater_back(grid, pos, direction) and is_valid_repeater_front(grid, pos, direction)


def validate_repeater_connections(
    grid: Dict[Tuple[int, int, int], str],
    pos: Tuple[int, int, int],
    direction: str,
    net_name: str = ""
) -> bool:
    """
    Validates that a repeater's front and back are connected to valid redstone.
    Prints warnings if invalid.
    
    Args:
        grid: The current block grid
        pos: Position of the repeater
        direction: Direction the repeater is facing (+x, -x, +z, -z)
        net_name: Name of the net (for warning message)
        
    Returns:
        True if both valid, False otherwise (also prints warnings)
    """
    valid = True
    
    if not is_valid_repeater_back(grid, pos, direction):
        back_pos = get_repeater_back_position(pos, direction)
        back_block = grid.get(back_pos) if back_pos else None
        print(f"Warning: Repeater at {pos} facing {direction} has invalid back connection. "
              f"Back position {back_pos} contains '{back_block}' (net: {net_name})")
        valid = False
    
    if not is_valid_repeater_front(grid, pos, direction):
        front_pos = get_repeater_front_position(pos, direction)
        front_block = grid.get(front_pos) if front_pos else None
        print(f"Warning: Repeater at {pos} facing {direction} has invalid front connection. "
              f"Front position {front_pos} contains '{front_block}' (net: {net_name})")
        valid = False
    
    return valid


def apply_signal_delay_repeaters(
    grid: Dict[Tuple[int, int, int], str], 
    routed_paths: Dict[str, List[List[Tuple[int, int, int]]]],
    wire_positions: Dict[Tuple[int, int, int], Dict[str, str]]
) -> Dict[Tuple[int, int, int], str]:
    """
    Post-processing pass to place repeaters for signal delay.
    Redstone signal dies after 15 blocks.
    """
    import collections
    
    for net_name, paths in routed_paths.items():
        if not paths:
            continue
            
        # Build graph for this net
        adj, source = build_net_graph(paths)
        if not source:
            continue
            
        # Traverse from source
        # State: (current_node, signal_strength, path_history)
        # path_history is a list of (node, is_safe_spot)
        
        queue = collections.deque([(source, 15)]) # Start with strength 15
        
        parents = {source: None}
        
        queue = collections.deque([source])
        signal_map = {source: 15}
        
        while queue:
            u = queue.popleft()
            
            current_signal = signal_map[u]
            
            if u not in adj:
                continue
                
            for v in adj[u]:
                # Each wire block reduces signal by 1, regardless of coordinate distance
                next_signal = current_signal - 1
                
                is_existing_repeater = False
                if v in grid and "repeater" in grid[v]:
                    is_existing_repeater = True
                    next_signal = 15
                
                if next_signal <= 0:
                    curr = u
                    safe_spot = None
                    safe_dir = None
                    
                    path_back = []
                    temp = curr
                    while temp:
                        path_back.append(temp)
                        temp = parents.get(temp)
                        if len(path_back) > 15:
                            break
                            
                    for i, cand in enumerate(path_back):
                        if grid.get(cand) != "redstone_wire":
                            continue
                            
                        if i == 0:
                            next_block = v
                        else:
                            next_block = path_back[i-1]
                            
                        dx = next_block[0] - cand[0]
                        dz = next_block[2] - cand[2]
                        
                        direction = None
                        if abs(dx) > abs(dz):
                            direction = '+x' if dx > 0 else '-x'
                        else:
                            direction = '+z' if dz > 0 else '-z'
                        
                        # Check if both front and back connections would be valid before placing
                        if not is_valid_repeater_placement(grid, cand, direction):
                            continue
                            
                        grid[cand] = f"repeater:{direction}"
                        
                        signal_map[cand] = 15
                        
                        queue.append(cand)
                        safe_spot = cand
                        break
                    
                    if not safe_spot:
                        print(f"Warning: Could not find safe repeater spot for net {net_name} near {u}")
                        pass
                    
                else:
                    # Signal is fine
                    if v not in signal_map or next_signal > signal_map[v]:
                        signal_map[v] = next_signal
                        parents[v] = u
                        queue.append(v)
                        
    return grid

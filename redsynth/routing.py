from typing import Dict, List, Tuple, Set, Optional
import heapq
import math
import time
from collections import deque
import networkx as nx

def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))

class RoutingGrid:
    def __init__(self, positions: Dict[str, Tuple[float, float, float]], nodes_data: Dict[str, dict]):
        self.positions = positions
        self.nodes_data = nodes_data
        self.blocked_coords = set()
        self.node_occupancy = {}
        self.node_to_coords = {} # name -> Set[coord]
        self.wire_occupancy = {} # (gx, gy, gz) -> Set[net_name]
        self.wire_directions = {} # (gx, gy, gz) -> Set[Tuple[int, int, int]]
        
        # Initialize bounds with infinity
        self.min_coords = [float('inf')] * 3
        self.max_coords = [float('-inf')] * 3
        
        self._build_grid()

    def _to_grid(self, val):
        return int(math.floor(val / 2))

    def _build_grid(self):
        # Calculate bounds in Grid Coordinates
        for pos in self.positions.values():
            for i in range(3):
                self.min_coords[i] = min(self.min_coords[i], pos[i])
                self.max_coords[i] = max(self.max_coords[i], pos[i])
        
        padding = 12
        self.min_coords = [self._to_grid(x - padding) for x in self.min_coords]
        self.max_coords = [self._to_grid(x + padding) for x in self.max_coords]

        print("Building routing grid obstacles (Voxel Mode)...")
        for name, pos in self.positions.items():
            dims = self.nodes_data[name].get('dims', (1, 1, 1))
            w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
            x, y, z = pos[0], pos[1], pos[2]
            
            # Pad the obstacle slightly in world space before converting
            pad = 0
            x_start = math.floor(x - w/2 - pad)
            x_end = math.ceil(x + w/2 + pad)
            y_start = math.floor(y - h/2 - pad)
            y_end = math.ceil(y + h/2 + pad)
            z_start = math.floor(z - d/2 - pad)
            z_end = math.ceil(z + d/2 + pad)
            
            # Mark all voxels that intersect with this volume
            for ix in range(x_start, x_end + 1):
                for iy in range(y_start, y_end + 1):
                    for iz in range(z_start, z_end + 1):
                        gx, gy, gz = self._to_grid(ix), self._to_grid(iy), self._to_grid(iz)
                        coord = (gx, gy, gz)
                        self.blocked_coords.add(coord)
                        self.node_occupancy[coord] = name
                        if name not in self.node_to_coords:
                            self.node_to_coords[name] = set()
                        self.node_to_coords[name].add(coord)

    def _is_cardinal(self, vec):
        return sum(1 for v in vec if v != 0) == 1

    def _is_perpendicular(self, v1, v2):
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] == 0

    def is_blocked(self, point, allowed_points=None, forceful=False):
        if point in self.blocked_coords:
            if allowed_points and point in allowed_points:
                return False
            
            if forceful:
                if point in self.node_occupancy:
                    return True
                return False # Bulldoze wires
                
            return True
        return False

    def get_node_coords(self, node_name):
        return self.node_to_coords.get(node_name, set())

    def get_segment_footprint(self, p1, p2):
        """Returns the set of coordinates occupied by the segment from p1 to p2."""
        footprint = set()
        footprint.add(p1)
        footprint.add(p2)
        
        x, y, z = p1
        nx, ny, nz = p2
        dy = ny - y
        dx = nx - x
        dz = nz - z

        # Vertical claim logic for diagonals
        # Claim the horizontal midpoint at both the start and end Y levels
        if abs(dy) > 0:
            mx = x + dx // 2
            mz = z + dz // 2
            # Claim at start Y level and end Y level
            footprint.add((mx, y, mz))
            footprint.add((mx, ny, mz))
        
        return footprint

    def add_path(self, path, net_name):
        for i in range(len(path)):
            p = path[i]
            self.blocked_coords.add(p)
            if p not in self.wire_occupancy:
                self.wire_occupancy[p] = set()
            self.wire_occupancy[p].add(net_name)

            if i < len(path) - 1:
                p1 = path[i]
                p2 = path[i+1]
                
                # Track directions
                dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                # We store the direction of the wire segment at both endpoints
                # p1 has outgoing vector (dx, dy, dz)
                # p2 has incoming vector (-dx, -dy, -dz)
                for pt, d in [(p1, (dx, dy, dz)), (p2, (-dx, -dy, -dz))]:
                    if pt not in self.wire_directions:
                        self.wire_directions[pt] = set()
                    self.wire_directions[pt].add(d)

                footprint = self.get_segment_footprint(p1, p2)
                for fp in footprint:
                    self.blocked_coords.add(fp)
                    if fp not in self.wire_occupancy:
                        self.wire_occupancy[fp] = set()
                    self.wire_occupancy[fp].add(net_name)

    def remove_path(self, path, net_name):
        for i in range(len(path)):
            p = path[i]
            
            if p in self.wire_occupancy:
                self.wire_occupancy[p].discard(net_name)
                if not self.wire_occupancy[p]:
                    del self.wire_occupancy[p]
                    self.blocked_coords.discard(p)
                    if p in self.wire_directions:
                        del self.wire_directions[p]
            
            if i < len(path) - 1:
                p1 = path[i]
                p2 = path[i+1]
                
                # Remove directions
                dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                for pt, d in [(p1, (dx, dy, dz)), (p2, (-dx, -dy, -dz))]:
                    if pt in self.wire_directions:
                        self.wire_directions[pt].discard(d)
                        if not self.wire_directions[pt]:
                            del self.wire_directions[pt]

                footprint = self.get_segment_footprint(p1, p2)
                for fp in footprint:
                    if fp in self.wire_occupancy:
                        self.wire_occupancy[fp].discard(net_name)
                        if not self.wire_occupancy[fp]:
                            del self.wire_occupancy[fp]
                            self.blocked_coords.discard(fp)

    def get_neighbors(self, point, allowed_points=None, forceful=False, prev_footprint=None, penalty_points=None, start_point=None, goal_point=None):
        x, y, z = point
        
        # New Move Set: Cardinals + Diagonals
        moves = []
        
        # Cardinals (Cost 1.0)
        cardinals = [
            (1, 0, 0), (-1, 0, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        for dx, dy, dz in cardinals:
            moves.append(((x+dx, y+dy, z+dz), 1.0))
            
        # Diagonals (Cost 1.414)
        # We only allow diagonals that change 2 coordinates (planar diagonals)
        # For redstone, usually we care about:
        # - Up/Down + Horizontal (Staircase) -> dy != 0 and (dx != 0 or dz != 0)
        slopes = [
            (2, 1, 0), (2, -1, 0),
            (-2, 1, 0), (-2, -1, 0),
            (0, 1, 2), (0, -1, 2),
            (0, 1, -2), (0, -1, -2)
        ]

        for dx, dy, dz in slopes:
            moves.append(((x+dx, y+dy, z+dz), 2.5))

        valid = []
        min_x, min_y, min_z = self.min_coords
        max_x, max_y, max_z = self.max_coords
        
        for (nx, ny, nz), cost in moves:
            if (min_x <= nx <= max_x and
                min_y <= ny <= max_y and
                min_z <= nz <= max_z):
                
                # Check destination
                if self.is_blocked((nx, ny, nz), allowed_points, forceful):
                    # Check for valid crossing
                    can_cross = False
                    if not forceful:
                        # Only allow crossing if blocked by WIRE, not NODE
                        if (nx, ny, nz) not in self.node_occupancy:
                             move_vec = (nx - x, ny - y, nz - z)
                             if self._is_cardinal(move_vec) and move_vec[1] == 0:
                                 # Check if we can cross INTO destination (perpendicular to dest wires)
                                 if (nx, ny, nz) in self.wire_directions:
                                     existing_dirs = self.wire_directions[(nx, ny, nz)]
                                     # Must be perpendicular to ALL existing directions at this point
                                     if existing_dirs and all(self._is_cardinal(d) and self._is_perpendicular(d, move_vec) for d in existing_dirs):
                                         can_cross = True
                                 elif (nx, ny, nz) in self.wire_occupancy:
                                     # Point has wire but no directions - it's a claimed point (port/footprint)
                                     # Allow crossing since we can't determine direction conflicts
                                     can_cross = True
                                 
                                 # Also allow crossing if we're EXITING perpendicular from current point
                                 # This handles the case where we're at a port that another wire crosses through
                                 if not can_cross and (x, y, z) in self.wire_directions:
                                     src_dirs = self.wire_directions[(x, y, z)]
                                     if src_dirs and all(self._is_cardinal(d) and self._is_perpendicular(d, move_vec) for d in src_dirs):
                                         can_cross = True
                    
                    if not can_cross:
                        continue
                
                # Check vertical claims for diagonals
                dy = ny - y
                dx = nx - x
                dz = nz - z
                if dy != 0:
                    # For slopes, we claim the horizontal midpoint at both start and end Y levels.
                    # Check both positions for blocking.
                    mx = x + dx // 2
                    mz = z + dz // 2
                    mid_at_start_y = (mx, y, mz)
                    mid_at_end_y = (mx, ny, mz)
                    
                    move_vec = (dx, dy, dz)
                    
                    # Check midpoint at start Y level
                    if self.is_blocked(mid_at_start_y, allowed_points, forceful):
                        # Allow if blocked by same-direction wire (parallel stacking)
                        is_compatible = False
                        if not forceful and mid_at_start_y in self.wire_directions:
                            if move_vec in self.wire_directions[mid_at_start_y]:
                                is_compatible = True
                        if not is_compatible:
                            continue
                    
                    # Check midpoint at end Y level
                    if self.is_blocked(mid_at_end_y, allowed_points, forceful):
                        # Allow if blocked by same-direction wire (parallel stacking)
                        is_compatible = False
                        if not forceful and mid_at_end_y in self.wire_directions:
                            if move_vec in self.wire_directions[mid_at_end_y]:
                                is_compatible = True
                        if not is_compatible:
                            continue
                
                # Apply penalty for bulldozing - count ALL unique nets in footprint
                final_cost = cost
                if forceful:
                    bulldozed_nets = set()
                    bulldozed_points = set()  # Track which points each net is bulldozed at
                    # Check destination
                    if (nx, ny, nz) in self.wire_occupancy:
                        bulldozed_nets.update(self.wire_occupancy[(nx, ny, nz)])
                        bulldozed_points.add((nx, ny, nz))
                    # Check full footprint for slope moves
                    if dy != 0:
                        footprint = self.get_segment_footprint((x, y, z), (nx, ny, nz))
                        for fp in footprint:
                            if fp in self.wire_occupancy:
                                bulldozed_nets.update(self.wire_occupancy[fp])
                                bulldozed_points.add(fp)
                    
                    # Calculate penalty - reduce for sloped wires and wires near endpoints
                    for net in bulldozed_nets:
                        is_sloped = False
                        is_near_endpoint = False
                        for bp in bulldozed_points:
                            if bp in self.wire_directions:
                                for d in self.wire_directions[bp]:
                                    if d[1] != 0:  # Has vertical component = sloped
                                        is_sloped = True
                                        break
                            # Check if within 1 block of start or goal
                            if start_point:
                                dist_to_start = abs(bp[0]-start_point[0]) + abs(bp[1]-start_point[1]) + abs(bp[2]-start_point[2])
                                if dist_to_start <= 2:
                                    is_near_endpoint = True
                            if goal_point:
                                dist_to_goal = abs(bp[0]-goal_point[0]) + abs(bp[1]-goal_point[1]) + abs(bp[2]-goal_point[2])
                                if dist_to_goal <= 2:
                                    is_near_endpoint = True
                        
                        # Base penalty 400, halved for slopes, halved again for near endpoints
                        base_penalty = 400.0
                        if is_sloped:
                            base_penalty *= 0.5
                        if is_near_endpoint:
                            base_penalty *= 0.1
                        final_cost += base_penalty

                # Apply penalty for differing from neighbors above/below
                move_vec = (nx - x, ny - y, nz - z)
                
                # Check Up
                up_pos = (x, y+1, z)
                if up_pos in self.wire_directions:
                     if move_vec not in self.wire_directions[up_pos]:
                         final_cost += 0.1
                         
                # Check Down
                down_pos = (x, y-1, z)
                if down_pos in self.wire_directions:
                     if move_vec not in self.wire_directions[down_pos]:
                         final_cost += 0.1

                # Apply penalty for traversing penalty points (node volumes)
                if penalty_points and (nx, ny, nz) in penalty_points:
                    final_cost += 800.0

                valid.append(((nx, ny, nz), final_cost))
                
        return valid


def a_star(start, goal, grid, allowed_points, max_steps=50_000, forceful=False, penalty_points=None, debug_goal=False, max_cost=None):
    def h(p1, p2):
        # Euclidean distance is better for 3D with diagonals
        return abs(p1[0]-p2[0]) + 2*abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        
    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    
    came_from = {}
    g_score = {start: 0}
    steps = 0
    
    # Track closest point for diagnostics
    closest_point = start
    closest_distance = h(start, goal)
    
    # Debug: track if goal was ever seen as a neighbor
    goal_seen_as_neighbor = False
    goal_added_to_open = False
    
    # Store best path found as fallback
    fallback_path = None
    fallback_cost = float('inf')
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        steps += 1
        
        # Update closest point tracking
        current_dist = h(current, goal)
        if current_dist < closest_distance:
            closest_distance = current_dist
            closest_point = current
        
        if steps > max_steps:
            # Return fallback path if we found one, otherwise return failure diagnostics
            if fallback_path is not None:
                return fallback_path
            return {
                'closest': closest_point,
                'distance': closest_distance,
                'steps': steps,
                'reason': 'max_steps',
                'goal_seen': goal_seen_as_neighbor,
                'goal_queued': goal_added_to_open
            }
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        prev_footprint = None
        prev_direction = None
        if current in came_from:
            previous = came_from[current]
            prev_footprint = grid.get_segment_footprint(previous, current)
            prev_direction = (current[0]-previous[0], current[1]-previous[1], current[2]-previous[2])
            
        for neighbor, move_cost in grid.get_neighbors(current, allowed_points, forceful, prev_footprint, penalty_points, start, goal):
            # Debug tracking
            if neighbor == goal:
                goal_seen_as_neighbor = True
            
            # Calculate turn penalty
            penalty = 0.0
            new_direction = (neighbor[0]-current[0], neighbor[1]-current[1], neighbor[2]-current[2])
            if prev_direction is not None and new_direction != prev_direction:
                penalty = 0.1
            
            # Check for illegal back moves with downward slopes
            # cardinal-cardinal back is allowed, but if either move is a downward slope, back is illegal
            if prev_direction is not None:
                prev_dx, prev_dy, prev_dz = prev_direction
                new_dx, new_dy, new_dz = new_direction
                
                # Check if moves go in opposite horizontal directions ("back")
                is_back = False
                if prev_dx != 0 and new_dx != 0 and (prev_dx > 0) != (new_dx > 0):
                    is_back = True
                if prev_dz != 0 and new_dz != 0 and (prev_dz > 0) != (new_dz > 0):
                    is_back = True
                
                # If going back and at least one move is a downward slope, skip this neighbor
                if is_back:
                    prev_is_downslope = prev_dy < 0
                    new_is_downslope = new_dy < 0
                    if prev_is_downslope or new_is_downslope:
                        continue

            tentative_g = current_g + move_cost + penalty
            
            # Skip if cost exceeds max_cost threshold
            if max_cost is not None and tentative_g > max_cost:
                continue
                
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                if neighbor == goal:
                    goal_added_to_open = True
                    # Store path as fallback if it's better than current best
                    if tentative_g < fallback_cost:
                        path = [goal]
                        node = current
                        while node in came_from:
                            path.append(node)
                            node = came_from[node]
                        path.append(start)
                        fallback_path = path[::-1]
                        fallback_cost = tentative_g
                
    # Return fallback path if we found one, otherwise return failure diagnostics
    if fallback_path is not None:
        return fallback_path
    return {
        'closest': closest_point,
        'distance': closest_distance,
        'steps': steps,
        'reason': 'exhausted',
        'goal_seen': goal_seen_as_neighbor,
        'goal_queued': goal_added_to_open
    }


def _analyze_blocking(grid, point, allowed_points=None):
    """Analyze what's blocking neighbors around a point."""
    x, y, z = point
    
    # Check all cardinal directions
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    node_blocked = 0
    node_blocked_but_allowed = 0
    wire_blocked = 0
    wire_nets = set()
    out_of_bounds = 0
    pin_blocked = 0
    blocking_nodes = set()
    
    min_x, min_y, min_z = grid.min_coords
    max_x, max_y, max_z = grid.max_coords
    
    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        
        # Check bounds
        if not (min_x <= nx <= max_x and min_y <= ny <= max_y and min_z <= nz <= max_z):
            out_of_bounds += 1
            continue
            
        neighbor = (nx, ny, nz)
        if neighbor in grid.node_occupancy:
            blocking_nodes.add(grid.node_occupancy[neighbor])
            if allowed_points and neighbor in allowed_points:
                node_blocked_but_allowed += 1
            else:
                node_blocked += 1
        elif neighbor in grid.wire_occupancy:
            wire_blocked += 1
            wire_nets.update(grid.wire_occupancy[neighbor])
        elif neighbor in grid.blocked_coords:
            # Blocked but not by node or wire - probably a pre-claimed pin
            pin_blocked += 1
    
    return {
        'node_blocked': node_blocked,
        'node_blocked_but_allowed': node_blocked_but_allowed,
        'wire_blocked': wire_blocked,
        'wire_nets': wire_nets,
        'out_of_bounds': out_of_bounds,
        'pin_blocked': pin_blocked,
        'blocking_nodes': blocking_nodes
    }


def _get_port_base(port_name):
    if '[' in port_name:
        return port_name.split('[')[0]
    return port_name

def _get_pin_pos(node_name, port_name, positions, nodes_data):
    pos = positions[node_name]
    pins = nodes_data[node_name]['pin_locations']
    dims = nodes_data[node_name]['dims']
    offset = pins.get(port_name, (0,0,0))
    # Calculate absolute position
    px = pos[0] - dims[0]/2 + offset[0]
    py = pos[1] - dims[1]/2 + offset[1]
    pz = pos[2] - dims[2]/2 + offset[2]
    return (px, py, pz)

def _is_8b_net(net_name, net_data, nodes_data):
    # Check driver
    driver_node, driver_port = net_data['driver']
    if driver_node and driver_node in nodes_data:
        driver_pins = nodes_data[driver_node]['pin_locations']
        base = _get_port_base(driver_port)
        # Count pins with this base
        count = sum(1 for p in driver_pins if _get_port_base(p) == base)
        if count >= 8:
            return True

    # Check sinks
    for sink_node, sink_port in net_data['sinks']:
        if sink_node and sink_node in nodes_data:
            sink_pins = nodes_data[sink_node]['pin_locations']
            base = _get_port_base(sink_port)
            count = sum(1 for p in sink_pins if _get_port_base(p) == base)
            if count >= 8:
                return True
    
    return False

def _get_net_length(net_data, positions, nodes_data):
    length = 0
    driver_node, driver_port = net_data['driver']
    if not driver_node: return float('inf')
    
    try:
        p1 = _get_pin_pos(driver_node, driver_port, positions, nodes_data)
        
        for sink_node, sink_port in net_data['sinks']:
            if sink_node:
                p2 = _get_pin_pos(sink_node, sink_port, positions, nodes_data)
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
                length += dist
    except KeyError:
        return float('inf')
        
    return length


def route_nets(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]], max_time_seconds: float = None) -> Tuple[Dict[str, List[List[Tuple[int, int, int]]]], List[Dict]]:
    print("Starting routing (Voxel Mode)...")
    
    nodes_data = {}
    for node in G.nodes():
        nodes_data[node] = {
            'dims': G.nodes[node].get('dims', (1, 1, 1)),
            'pin_locations': G.nodes[node].get('pin_locations', {})
        }
        
    grid = RoutingGrid(positions, nodes_data)
    
    # Validate pin spacing
    print("Validating pin spacing...")
    all_pins = []
    for node, pos in positions.items():
        pins = nodes_data[node]['pin_locations']
        dims = nodes_data[node]['dims']
        for pin_name, pin_offset in pins.items():
            # Calculate absolute pin position
            px = pos[0] - dims[0]/2 + pin_offset[0]
            py = pos[1] - dims[1]/2 + pin_offset[1]
            pz = pos[2] - dims[2]/2 + pin_offset[2]
            all_pins.append(((px, py, pz), f"{node}.{pin_name}"))

    for i in range(len(all_pins)):
        p1, name1 = all_pins[i]
        for j in range(i + 1, len(all_pins)):
            p2, name2 = all_pins[j]
            dist_sq = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2
            if dist_sq < 4: # Distance < 2 means dist_sq < 4
                raise ValueError(f"Pin spacing violation: {name1} and {name2} are too close (dist_sq={dist_sq}). Positions: {p1}, {p2}")
    
    # Pre-claim all pin locations to prevent other nets from routing through them
    print("Pre-claiming pin locations...")
    pin_reservations = {}  # grid_coord -> set of "node.pin" names
    for (px, py, pz), pin_name in all_pins:
        gp = (grid._to_grid(px), grid._to_grid(py), grid._to_grid(pz))
        if gp not in pin_reservations:
            pin_reservations[gp] = set()
        pin_reservations[gp].add(pin_name)
        # Mark as blocked so routing avoids these points
        grid.blocked_coords.add(gp)
    
    print(f"  Reserved {len(pin_reservations)} unique pin grid locations")
    
    nets = {}
    for u, v, data in G.edges(data=True):
        net_name = data.get('label', 'unknown')
        src_port = data.get('src_port')
        dst_port = data.get('dst_port')
        
        if net_name not in nets:
            nets[net_name] = {'driver': None, 'sinks': []}
        
        if not nets[net_name]['driver']:
             nets[net_name]['driver'] = (u, src_port)
        
        nets[net_name]['sinks'].append((v, dst_port))
        
    routed_paths = {}
    failed_connections_list = []
    
    failed_connections = 0
    total_connections = 0
    
    # Sort nets by priority: 8b first, then shortest length
    print("Prioritizing nets...")
    net_priorities = []
    for net_name, net_data in nets.items():
        is_8b = _is_8b_net(net_name, net_data, nodes_data)
        length = _get_net_length(net_data, positions, nodes_data)
        # Priority: 8b first (True > False), then length (Short > Long)
        # Sort key: (not is_8b, length)
        # False < True, so (False, len) comes before (True, len) -> 8b comes first
        net_priorities.append((net_name, (not is_8b, length)))
    
    net_priorities.sort(key=lambda x: x[1])
    sorted_nets = [n[0] for n in net_priorities]
    
    # Queue of nets to route
    routing_queue = deque(sorted_nets)
    rip_up_counts = {net: 0 for net in nets}
    MAX_RIP_UPS = 10
    
    start_time = time.time()
    
    while routing_queue:
        print(f"{len(routing_queue)}/{len(nets)} nets left to route.\r", end="")
        if max_time_seconds and (time.time() - start_time > max_time_seconds):
            print(f"Stopping routing after {max_time_seconds} seconds.")
            break

        net_name = routing_queue.popleft()
        net_data = nets[net_name]
        
        driver_node, driver_port = net_data['driver']
        sinks = net_data['sinks']
        
        if not driver_node or not sinks:
            continue
            
        total_connections += len(sinks)
        
        # Calculate Driver Grid Point
        d_pos = positions[driver_node]
        d_pins = nodes_data[driver_node]['pin_locations']
        d_pin_offset = d_pins.get(driver_port, (0, 0, 0))
        d_dims = nodes_data[driver_node]['dims']
        
        dx = d_pos[0] - d_dims[0]/2 + d_pin_offset[0]
        dy = d_pos[1] - d_dims[1]/2 + d_pin_offset[1]
        dz = d_pos[2] - d_dims[2]/2 + d_pin_offset[2]
        
        start_point = (grid._to_grid(dx), grid._to_grid(dy), grid._to_grid(dz))
        
        # Initialize tree
        tree_points = {start_point}
        
        # Allow breakout from driver
        # We don't add to tree_points, but we will add to allowed_points during A*
        pass

        remaining_sinks = []
        sink_breakouts = {} # Map sink point to set of allowed breakout points

        for idx, (sink_node, sink_port) in enumerate(sinks):
            s_pos = positions[sink_node]
            s_pins = nodes_data[sink_node]['pin_locations']
            s_pin_offset = s_pins.get(sink_port, (0, 0, 0))
            s_dims = nodes_data[sink_node]['dims']
            
            sx = s_pos[0] - s_dims[0]/2 + s_pin_offset[0]
            sy = s_pos[1] - s_dims[1]/2 + s_pin_offset[1]
            sz = s_pos[2] - s_dims[2]/2 + s_pin_offset[2]
            
            sp = (grid._to_grid(sx), grid._to_grid(sy), grid._to_grid(sz))
            remaining_sinks.append(sp)
            
            # Ensure sink point is claimed
            if sp not in grid.blocked_coords:
                grid.blocked_coords.add(sp)
                if sp not in grid.wire_occupancy:
                    grid.wire_occupancy[sp] = set()
                grid.wire_occupancy[sp].add(net_name)
            
            # Allow breakout from sink
            sink_allowed = set()
            if sink_node:
                sink_allowed.update(grid.get_node_coords(sink_node))
            sink_breakouts[sp] = sink_allowed

        current_net_paths = []
        success = True
        
        while remaining_sinks:
            best_dist = float('inf')
            best_pair = None
            
            for sp in remaining_sinks:
                for tp in tree_points:
                     dist = (tp[0]-sp[0])**2 + (tp[1]-sp[1])**2 + (tp[2]-sp[2])**2
                     if dist < best_dist:
                         best_dist = dist
                         best_pair = (tp, sp)
            
            if not best_pair:
                success = False
                break
                
            start, end = best_pair
            allowed_points = set(tree_points)
            allowed_points.add(end)
            penalty_points = set()
            
            if driver_node:
                driver_coords = grid.get_node_coords(driver_node)
                allowed_points.update(driver_coords)
                penalty_points.update(driver_coords)
                
            if end in sink_breakouts:
                sink_coords = sink_breakouts[end]
                allowed_points.update(sink_coords)
                penalty_points.update(sink_coords)
            
            # Remove start and end from penalty points so we don't penalize connecting to the pin itself
            if start in penalty_points:
                penalty_points.remove(start)
            if end in penalty_points:
                penalty_points.remove(end)
            
            result = a_star(start, end, grid, allowed_points, forceful=False, penalty_points=penalty_points)
            
            # Track failure diagnostics for reporting
            failure_diag = None
            if isinstance(result, dict):
                failure_diag = result
                path = None
            else:
                path = result
            
            if path is None and rip_up_counts[net_name] < MAX_RIP_UPS:
                 result = a_star(start, end, grid, allowed_points, max_steps=100_000, forceful=True, penalty_points=penalty_points, max_cost=800)
                 if isinstance(result, dict):
                     failure_diag = result
                     path = None
                 else:
                     path = result
                 if path:
                    # Handle rip-up
                    collided_nets = set()
                    for i in range(len(path)):
                        p = path[i]
                        if p in grid.wire_occupancy:
                            for victim in grid.wire_occupancy[p]:
                                if victim != net_name:
                                    collided_nets.add(victim)
                        # Check vertical claims too?
                        if i < len(path) - 1:
                            footprint = grid.get_segment_footprint(p, path[i+1])
                            for fp in footprint:
                                if fp in grid.wire_occupancy:
                                    for victim in grid.wire_occupancy[fp]:
                                        if victim != net_name:
                                            collided_nets.add(victim)

                    if collided_nets:
                        print(f"  [Rip-up] Net {net_name} ripping up: {collided_nets}")
                        for victim in collided_nets:
                            if victim in routed_paths:
                                for vp in routed_paths[victim]:
                                    grid.remove_path(vp, victim)
                                del routed_paths[victim]
                                routing_queue.append(victim)
                                rip_up_counts[victim] += 1

            if path:
                current_net_paths.append(path)
                grid.add_path(path, net_name)
                for p in path:
                    tree_points.add(p)
                remaining_sinks.remove(end)
            else:
                # Print detailed failure diagnostics
                print(f"\n  [Fail] Net '{net_name}' from {start} to {end}")
                if failure_diag:
                    closest = failure_diag['closest']
                    dist = failure_diag['distance']
                    steps = failure_diag['steps']
                    reason = failure_diag['reason']
                    goal_seen = failure_diag.get('goal_seen', False)
                    goal_queued = failure_diag.get('goal_queued', False)
                    print(f"         Closest point: {closest}, distance remaining: {dist:.2f}")
                    print(f"         Reason: {reason} after {steps} steps")
                    print(f"         Goal seen as neighbor: {goal_seen}, Goal added to queue: {goal_queued}")
                    
                    # Analyze blocking near closest point
                    blocking = _analyze_blocking(grid, closest, allowed_points)
                    node_b = blocking['node_blocked']
                    node_allowed = blocking['node_blocked_but_allowed']
                    wire_b = blocking['wire_blocked']
                    oob = blocking['out_of_bounds']
                    pin_b = blocking['pin_blocked']
                    wire_nets = blocking['wire_nets']
                    blocking_nodes = blocking['blocking_nodes']
                    
                    parts = []
                    if node_b > 0:
                        nodes_str = ", ".join(list(blocking_nodes)[:3])
                        if len(blocking_nodes) > 3:
                            nodes_str += f" +{len(blocking_nodes)-3} more"
                        parts.append(f"{node_b} by other nodes ({nodes_str})")
                    if node_allowed > 0:
                        parts.append(f"{node_allowed} by own node (allowed)")
                    if wire_b > 0:
                        nets_str = ", ".join(list(wire_nets)[:3])
                        if len(wire_nets) > 3:
                            nets_str += f" +{len(wire_nets)-3} more"
                        parts.append(f"{wire_b} by wires ({nets_str})")
                    if pin_b > 0:
                        parts.append(f"{pin_b} by other pins")
                    if oob > 0:
                        parts.append(f"{oob} out of bounds")
                    
                    if parts:
                        print(f"         Blocking: {', '.join(parts)}")
                    
                    # Extra diagnostics: why can't we reach the goal from closest?
                    if dist < 2.0:
                        print(f"         --- Goal reachability analysis ---")
                        print(f"         Goal in blocked_coords: {end in grid.blocked_coords}")
                        print(f"         Goal in node_occupancy: {end in grid.node_occupancy}")
                        print(f"         Goal in wire_occupancy: {grid.wire_occupancy.get(end, set())}")
                        print(f"         Goal in allowed_points: {end in allowed_points}")
                        print(f"         Tree points near goal: {[tp for tp in tree_points if abs(tp[0]-end[0]) + abs(tp[1]-end[1]) + abs(tp[2]-end[2]) <= 2]}")
                        
                        # Check what moves from closest would reach goal
                        dx, dy, dz = end[0] - closest[0], end[1] - closest[1], end[2] - closest[2]
                        print(f"         Move needed: ({dx}, {dy}, {dz})")
                        
                        # Check if that move is in the move set
                        cardinals = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
                        slopes = [(2,1,0), (2,-1,0), (-2,1,0), (-2,-1,0), (0,1,2), (0,-1,2), (0,1,-2), (0,-1,-2)]
                        all_moves = cardinals + slopes
                        if (dx, dy, dz) in all_moves:
                            print(f"         Move IS in move set")
                        else:
                            print(f"         Move NOT in move set! Available: cardinals + slopes")
                
                failed_connections += 1
                failed_connections_list.append({'net': net_name, 'start': start, 'end': end})
                remaining_sinks.remove(end)
                print(f"{len(routing_queue)}/{len(nets)} nets left to route.\r", end="")
                success = False
        
        # Convert paths back to world coordinates (x2) for output
        # Actually, let's return the grid coordinates and let redstone.py handle the scaling/filling
        # But wait, visualization might expect world coords?
        # The plan said: "Return the list of 'Key Points' (Grid Coords * 2)"
        
        world_paths = []
        for path in current_net_paths:
            world_path = [(p[0]*2, p[1]*2, p[2]*2) for p in path]
            world_paths.append(world_path)
            
        routed_paths[net_name] = world_paths

    print(f"\nRouting complete. Failed connections: {failed_connections}/{total_connections}")
    return routed_paths, failed_connections_list

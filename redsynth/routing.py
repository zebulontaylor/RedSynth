from typing import Dict, List, Tuple, Set, Optional
import heapq
import math
import time
from collections import deque
import networkx as nx

class RoutingGrid:
    def __init__(self, positions: Dict[str, Tuple[float, float, float]], nodes_data: Dict[str, dict]):
        self.positions = positions
        self.nodes_data = nodes_data
        self.blocked_coords = set()
        self.node_occupancy = {}
        self.wire_occupancy = {} # (x,y,z) -> net_name
        self.min_coords = [float('inf')] * 3
        self.max_coords = [float('-inf')] * 3
        self._build_grid()

    def _build_grid(self):
        for pos in self.positions.values():
            for i in range(3):
                self.min_coords[i] = min(self.min_coords[i], pos[i])
                self.max_coords[i] = max(self.max_coords[i], pos[i])
        
        padding = 10
        self.min_coords = [int(x - padding) for x in self.min_coords]
        self.max_coords = [int(x + padding) for x in self.max_coords]

        print("Building routing grid obstacles...")
        for name, pos in self.positions.items():
            dims = self.nodes_data[name].get('dims', (1, 1, 1))
            w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
            x, y, z = pos[0], pos[1], pos[2]
            
            pad = 1
            x_start = math.floor(x - w/2 - pad)
            x_end = math.ceil(x + w/2 + pad)
            y_start = math.floor(y - h/2 - pad)
            y_end = math.ceil(y + h/2 + pad)
            z_start = math.floor(z - d/2 - pad)
            z_end = math.ceil(z + d/2 + pad)
            
            for ix in range(x_start, x_end + 1):
                for iy in range(y_start, y_end + 1):
                    for iz in range(z_start, z_end + 1):
                        coord = (ix, iy, iz)
                        self.blocked_coords.add(coord)
                        self.node_occupancy[coord] = name

    def is_blocked(self, point, allowed_points=None, forceful=False):
        if point in self.blocked_coords:
            if allowed_points and point in allowed_points:
                return False
            
            # If forceful, only hard obstacles (nodes) block us. Wires don't.
            if forceful:
                if point in self.node_occupancy:
                    return True
                return False # It's a wire, so we can bulldoze (at a cost)
                
            return True
        return False

    def add_path(self, path, net_name):
        for i in range(len(path)):
            p = path[i]
            self.blocked_coords.add(p)
            self.wire_occupancy[p] = net_name

            if i < len(path) - 1:
                p1 = path[i]
                p2 = path[i+1]
                
                # Check for vertical slope
                if abs(p2[1] - p1[1]) == 1:
                    x, y, z = p1
                    nx, ny, nz = p2
                    dx = nx - x
                    dy = ny - y
                    dz = nz - z
                    
                    mid1 = (x + dx//2, y, z + dz//2)
                    mid2 = (x + dx//2, y + dy, z + dz//2)
                    self.blocked_coords.add(mid1)
                    self.blocked_coords.add(mid2)
                    self.wire_occupancy[mid1] = net_name
                    self.wire_occupancy[mid2] = net_name

                    # Claim support for downward moves
                    if dy == -1:
                        mid_support = (mid2[0], mid2[1]-1, mid2[2])
                        self.blocked_coords.add(mid_support)
                        self.wire_occupancy[mid_support] = net_name

    def remove_path(self, path):
        """Removes a path from the grid (rip-up)."""
        for i in range(len(path)):
            p = path[i]
            self.blocked_coords.discard(p)
            self.wire_occupancy.pop(p, None)

            if i < len(path) - 1:
                p1 = path[i]
                p2 = path[i+1]
                
                if abs(p2[1] - p1[1]) == 1:
                    x, y, z = p1
                    nx, ny, nz = p2
                    dx = nx - x
                    dy = ny - y
                    dz = nz - z
                    
                    mid1 = (x + dx//2, y, z + dz//2)
                    mid2 = (x + dx//2, y + dy, z + dz//2)
                    self.blocked_coords.discard(mid1)
                    self.blocked_coords.discard(mid2)
                    self.wire_occupancy.pop(mid1, None)
                    self.wire_occupancy.pop(mid2, None)

                    # Release support for downward moves
                    if dy == -1:
                        mid_support = (mid2[0], mid2[1]-1, mid2[2])
                        self.blocked_coords.discard(mid_support)
                        self.wire_occupancy.pop(mid_support, None)

    def get_neighbors(self, point, allowed_points=None, forceful=False):
        x, y, z = point
        moves = [
            # Horizontal moves (slope 0) - Cost 1
            ((x+1, y, z), 1.0), ((x-1, y, z), 1.0),
            ((x, y, z+1), 1.0), ((x, y, z-1), 1.0),
            # Going through a wire - Cost 1.2 (prefer straight shots over individual horizontal movements)
            ((x+2, y, z), 1.2), ((x-2, y, z), 1.2),
            ((x, y, z+2), 1.2), ((x, y, z-2), 1.2),
            # Sloped vertical moves (slope 1/2) - Cost 3 (approx Manhattan dist)
            ((x+2, y+1, z), 3.0), ((x-2, y+1, z), 3.0),
            ((x, y+1, z+2), 3.0), ((x, y+1, z-2), 3.0),
            ((x+2, y-1, z), 3.0), ((x-2, y-1, z), 3.0),
            ((x, y-1, z+2), 3.0), ((x, y-1, z-2), 3.0)
        ]
        valid = []
        min_x, min_y, min_z = self.min_coords
        max_x, max_y, max_z = self.max_coords
        
        for (nx, ny, nz), cost in moves:
            if (min_x <= nx <= max_x and
                min_y <= ny <= max_y and
                min_z <= nz <= max_z):
                
                if self.is_blocked((nx, ny, nz), allowed_points, forceful):
                    continue
                # Wires are 2 tall.
                # If the target point is an allowed point (e.g. a pin or existing wire), we skip the height check
                # to allow connecting to pins that might be flush against a component, or traversing existing wires.
                is_target = allowed_points is not None and (nx, ny, nz) in allowed_points

                dx = nx - x
                dy = ny - y
                dz = nz - z
                
                if not is_target:
                    if self.is_blocked((nx, ny-1, nz), allowed_points, forceful):
                        continue
                    # Check sides (perpendicular) to prevent crosstalk
                    # If moving in X (dx=1, dz=0), check Z neighbors (offset by dx)
                    # If moving in Z (dx=0, dz=1), check X neighbors (offset by dz)
                    if self.is_blocked((nx+dz, ny, nz+dx), allowed_points, forceful):
                        continue
                    if self.is_blocked((nx-dz, ny, nz-dx), allowed_points, forceful):
                        continue

                # Check for clipping on vertical moves
                if abs(dy) == 1:
                    mid1 = (x + dx//2, y, z + dz//2)
                    mid2 = (x + dx//2, y + dy, z + dz//2)
                    
                    if (self.is_blocked(mid1, allowed_points, forceful) or self.is_blocked(mid2, allowed_points, forceful)):
                        continue

                    # Check support for downward moves (mid2 needs support at y-2)
                    if dy == -1:
                        mid_support = (mid2[0], mid2[1]-1, mid2[2])
                        if self.is_blocked(mid_support, allowed_points, forceful):
                            continue

                    # Check crosstalk for intermediate wire (mid2)
                    # Perpendicular offsets: if moving in X, check Z neighbors.
                    p_dx = dz // 2
                    p_dz = dx // 2
                    
                    if self.is_blocked((mid2[0]+p_dx, mid2[1], mid2[2]+p_dz), allowed_points, forceful):
                        continue
                    if self.is_blocked((mid2[0]-p_dx, mid2[1], mid2[2]-p_dz), allowed_points, forceful):
                        continue

                # Apply penalty for bulldozing
                final_cost = cost
                if forceful:
                    # Check if we are stepping on a wire (soft block)
                    # We check the point and the point above it (since wires are 2 tall)
                    is_colliding = False
                    if (nx, ny, nz) in self.wire_occupancy: is_colliding = True
                    if (nx, ny+1, nz) in self.wire_occupancy: is_colliding = True
                    
                    if is_colliding:
                        final_cost += 20.0 # Penalty for ripping up a net (reduced to avoid A* timeout)

                valid.append(((nx, ny, nz), final_cost))
        return valid


def a_star(start, goal, grid, allowed_points, max_steps=50000, forceful=False):
    def h(p1, p2):
        # Standard Manhattan distance
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        
    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    
    came_from = {}
    g_score = {start: 0}
    steps = 0
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        steps += 1
        if steps > max_steps:
            print(f"A* max steps reached ({max_steps}) from {start} to {goal}")
            return None
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        for neighbor, move_cost in grid.get_neighbors(current, allowed_points, forceful):
            if grid.is_blocked(neighbor, allowed_points, forceful):
                continue
                
            tentative_g = current_g + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                
    return None


def find_tunnel(grid, start_point):
    """Finds the shortest path from start_point to any non-blocked point using only cardinal directions."""
    if start_point not in grid.blocked_coords:
        return []

    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    valid_paths = []
    
    for dx, dy, dz in directions:
        path = [start_point]
        curr_x, curr_y, curr_z = start_point
        
        # Limit tunnel length to avoid infinite loops or extremely long paths
        max_tunnel_len = 10
        
        for _ in range(max_tunnel_len):
            curr_x += dx
            curr_y += dy
            curr_z += dz
            next_point = (curr_x, curr_y, curr_z)
            path.append(next_point)
            
            if not grid.is_blocked(next_point):
                valid_paths.append(path)
                break
                
    if not valid_paths:
        return []
        
    # Return the shortest path
    return min(valid_paths, key=len)


def route_nets(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]], max_time_seconds: float = None) -> Tuple[Dict[str, List[List[Tuple[int, int, int]]]], List[Dict]]:
    print("Starting routing...")
    
    nodes_data = {}
    for node in G.nodes():
        nodes_data[node] = {
            'dims': G.nodes[node].get('dims', (1, 1, 1)),
            'pin_locations': G.nodes[node].get('pin_locations', {})
        }
        
    grid = RoutingGrid(positions, nodes_data)
    
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
    
    # --- Pass 1: Pre-calculate tunnels and claim endpoints ---
    print("Pass 1: Running tunneling for all pins...")
    
    # Store tunnels and points: 
    # net_name -> { 
    #   'driver': {'point': (x,y,z), 'tunnel': path}, 
    #   'sinks': { sink_idx: {'point': (x,y,z), 'tunnel': path} } 
    # }
    precalculated_data = {} 
    
    total_nets = len(nets)
    
    start_time = time.time()

    for i, (net_name, net_data) in enumerate(nets.items()):
        driver_node, driver_port = net_data['driver']
        sinks = net_data['sinks']
        
        if not driver_node or not sinks:
            continue

        precalculated_data[net_name] = {'driver': None, 'sinks': {}}
            
        # 1. Driver Tunnel
        d_pos = positions[driver_node]
        d_pins = nodes_data[driver_node]['pin_locations']
        d_pin_offset = d_pins.get(driver_port, (0, 0, 0))
        d_dims = nodes_data[driver_node]['dims']
        
        start_x = int(round(d_pos[0] - d_dims[0]/2 + d_pin_offset[0]))
        start_y = int(round(d_pos[1] - d_dims[1]/2 + d_pin_offset[1]))
        start_z = int(round(d_pos[2] - d_dims[2]/2 + d_pin_offset[2]))
        start_point = (start_x, start_y, start_z)
        
        driver_tunnel = find_tunnel(grid, start_point)
        precalculated_data[net_name]['driver'] = {'point': start_point, 'tunnel': driver_tunnel}
        
        if driver_tunnel:
            # Claim the tunnel immediately
            for p in driver_tunnel:
                grid.blocked_coords.add(p)
        elif start_point in grid.blocked_coords:
             pass
        else:
            grid.blocked_coords.add(start_point)

        # 2. Sink Tunnels
        for idx, (sink_node, sink_port) in enumerate(sinks):
            s_pos = positions[sink_node]
            s_pins = nodes_data[sink_node]['pin_locations']
            s_pin_offset = s_pins.get(sink_port, (0, 0, 0))
            s_dims = nodes_data[sink_node]['dims']
            
            sx = int(round(s_pos[0] - s_dims[0]/2 + s_pin_offset[0]))
            sy = int(round(s_pos[1] - s_dims[1]/2 + s_pin_offset[1]))
            sz = int(round(s_pos[2] - s_dims[2]/2 + s_pin_offset[2]))
            
            p = (sx, sy, sz)
            
            sink_tunnel = find_tunnel(grid, p)
            precalculated_data[net_name]['sinks'][idx] = {'point': p, 'tunnel': sink_tunnel}
            
            if sink_tunnel:
                for tp in sink_tunnel:
                    grid.blocked_coords.add(tp)
            elif p in grid.blocked_coords:
                pass
            else:
                grid.blocked_coords.add(p)

    # --- Pass 2: Main Routing with Rip-up and Reroute ---
    print("Pass 2: connecting nets with Rip-up and Reroute...")
    
    # Queue of nets to route: (priority, net_name)
    # We use a priority queue to prioritize nets that have been ripped up less often?
    # Or just a simple deque. Let's use a deque for simplicity first.
    routing_queue = deque(nets.keys())
    
    # Track how many times each net has been ripped up
    rip_up_counts = {net: 0 for net in nets}
    MAX_RIP_UPS = 10
    
    while routing_queue:
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
        
        # Retrieve pre-calculated data
        net_precalc = precalculated_data.get(net_name, {})
        driver_info = net_precalc.get('driver')
        if not driver_info:
            continue
            
        start_point = driver_info['point']
        start_tunnel = driver_info['tunnel']
        
        # Initialize tree with start point and tunnel
        tree_points = {start_point}
        for p in start_tunnel:
            tree_points.add(p)

        remaining_sinks = []
        sink_tunnels = {} # Local map for this net: point -> tunnel
        
        for idx, (sink_node, sink_port) in enumerate(sinks):
            sink_info = net_precalc.get('sinks', {}).get(idx)
            if not sink_info:
                continue
                
            p = sink_info['point']
            tun = sink_info['tunnel']
            
            remaining_sinks.append(p)
            sink_tunnels[p] = tun
            
        net_paths = []
        success = True
        
        # Temporary storage for paths in this iteration
        current_net_paths = []
        
        while remaining_sinks:
            best_dist = float('inf')
            best_pair = None
            
            for sp in remaining_sinks:
                for tp in tree_points:
                     dist = abs(tp[0]-sp[0]) + abs(tp[1]-sp[1]) + abs(tp[2]-sp[2])
                     if dist < best_dist:
                         best_dist = dist
                         best_pair = (tp, sp)
            
            if not best_pair:
                success = False
                break
                
            start, end = best_pair
            
            allowed_points = set(tree_points)
            
            if end in sink_tunnels:
                for p in sink_tunnels[end]:
                    allowed_points.add(p)
            allowed_points.add(end) 
            
            # Try normal routing first
            path = a_star(start, end, grid, allowed_points, forceful=False)
            
            if not path:
                # Normal routing failed. Try forceful routing if we haven't exceeded rip-up limit.
                if rip_up_counts[net_name] < MAX_RIP_UPS:
                    # print(f"  [Force] Net {net_name} needs to bulldoze...")
                    # Increase max_steps for forceful search to ensure we find a path even if it's long
                    path = a_star(start, end, grid, allowed_points, max_steps=100000, forceful=True)
                    
                    if path:
                        # Identify collisions
                        collided_nets = set()
                        for p in path:
                            # Check p and p+1 (wire height)
                            if p in grid.wire_occupancy:
                                collided_nets.add(grid.wire_occupancy[p])
                            p_up = (p[0], p[1]+1, p[2])
                            if p_up in grid.wire_occupancy:
                                collided_nets.add(grid.wire_occupancy[p_up])
                        
                        # Filter out self (shouldn't happen if logic is correct, but safe to check)
                        if net_name in collided_nets:
                            collided_nets.remove(net_name)
                            
                        # Rip up collided nets
                        if collided_nets:
                            print(f"  [Rip-up] Net {net_name} ripping up: {collided_nets}")
                            for victim in collided_nets:
                                if victim in routed_paths:
                                    # Remove victim's paths from grid
                                    for vp in routed_paths[victim]:
                                        grid.remove_path(vp)
                                    # Remove from routed_paths
                                    del routed_paths[victim]
                                    # Add back to queue
                                    routing_queue.append(victim)
                                    rip_up_counts[victim] += 1
                
            if path:
                current_net_paths.append(path)
                # We don't add to grid yet? No, we must add to grid immediately so subsequent segments of THIS net respect it.
                # But if we fail later in this net, we might need to rollback?
                # For now, let's assume we commit segment by segment.
                # Wait, if we are in forceful mode, we might be stepping on nets we just ripped up?
                # No, we ripped them up, so they are gone from grid.
                
                grid.add_path(path, net_name)
                for p in path:
                    tree_points.add(p)
                
                if end in sink_tunnels:
                    for p in sink_tunnels[end]:
                        tree_points.add(p)
                        
                remaining_sinks.remove(end)
            else:
                print(f"  [Fail] No path found for net {net_name} from {start} to {end}")
                failed_connections += 1
                failed_connections_list.append({
                    'net': net_name,
                    'start': start,
                    'end': end
                })
                remaining_sinks.remove(end)
                success = False
        
        # Add the used portion of the driver tunnel to the routed paths
        if driver_tunnel:
            driver_tunnel_indices = {p: i for i, p in enumerate(driver_tunnel)}
            max_driver_idx = -1
            
            # Check all points in the routed paths to see how far up the tunnel we went
            for path in current_net_paths:
                if path and path[0] in driver_tunnel_indices:
                    idx = driver_tunnel_indices[path[0]]
                    max_driver_idx = max(max_driver_idx, idx)
                if path and path[-1] in driver_tunnel_indices:
                    idx = driver_tunnel_indices[path[-1]]
                    max_driver_idx = max(max_driver_idx, idx)
            
            # Also check the start point itself (index 0 of tunnel usually)
            # Actually, the tunnel starts at index 0 near the driver.
            # If we used any point in the tunnel, we need the segment from 0 to that point.
            
            if max_driver_idx >= 0:
                used_tunnel = driver_tunnel[:max_driver_idx+1]
                # Only add if it has length (more than 1 point or just 1 point? A path usually needs 2 points to be a line)
                # But a single point path is valid for connectivity if it overlaps.
                # Let's add it if it's not empty.
                if used_tunnel:
                    # We need to make sure we don't duplicate paths if they are already covered?
                    # But the tunnel is a specific pre-calculated path.
                    # Let's add it.
                    current_net_paths.append(used_tunnel)
                    grid.add_path(used_tunnel, net_name)
                
        if success:
            routed_paths[net_name] = current_net_paths
        else:
            # If we failed partially, we should probably clean up what we placed for this net?
            # Or just leave it as a partial route.
            # Let's leave it for now, or maybe rip it up so it doesn't block others?
            # If it failed, it's likely stuck. Leaving it might be better than nothing.
            routed_paths[net_name] = current_net_paths
        
    print(f"\nRouting complete. Failed connections: {failed_connections}/{total_connections}")
    
    return routed_paths, failed_connections_list

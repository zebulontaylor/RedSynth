#!/usr/bin/env python3
"""
Netlist parser for Yosys JSON output.
"""
import json
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import heapq
import random
try:
    import plotly.graph_objects as go
except ImportError:
    go = None



class Node:
    """Represents a node in the netlist with bounding box and connections."""
    
    def __init__(self, name: str, cell_type: str = None):
        self.name = name
        self.cell_type = cell_type
        self.bounding_box: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
        self.incoming: Dict[str, List[str]] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.pin_locations: Dict[str, Tuple[int, int, int]] = {}
    
    def set_bounding_box(self, x: int, y: int, z: int, width: int, height: int, depth: int):
        self.bounding_box = (x, y, z, width, height, depth)
    
    def add_incoming_connection(self, port: str, connection_key: str):
        if port not in self.incoming:
            self.incoming[port] = []
        self.incoming[port].append(connection_key)
    
    def add_outgoing_connection(self, port: str, connection_key: str):
        if port not in self.outgoing:
            self.outgoing[port] = []
        self.outgoing[port].append(connection_key)
        
    def set_pin_location(self, port: str, x: int, y: int, z: int):
        self.pin_locations[port] = (x, y, z)
    
    def __repr__(self):
        return (f"Node(name={self.name!r}, type={self.cell_type!r}, "
                f"bbox={self.bounding_box}, "
                f"in={list(self.incoming.keys())}, "
                f"out={list(self.outgoing.keys())})")


def parse_netlist(json_file_path: str, cells_json_path: str = "cells/cells.json") -> List[Node]:
    """
    Parse a Yosys netlist JSON file and return a list of Node models.
    
    Args:
        json_file_path: Path to the netlist JSON file
        cells_json_path: Path to the cells layout JSON file
        
    Returns:
        List of Node objects representing the netlist
    """
    with open(json_file_path, 'r') as f:
        netlist = json.load(f)
    
    # Load cell layout information
    cell_layouts = {}
    with open(cells_json_path, 'r') as f:
        cell_layouts = json.load(f)
    
    nodes = []
    
    for module_name, module_data in netlist.get('modules', {}).items():
        cells = module_data.get('cells', {})
        ports = module_data.get('ports', {})
        
        if not cells and not ports:
            continue

        if cells:
            for port_name, port_data in ports.items():
                direction = port_data.get('direction', 'input')
                bits = port_data.get('bits', [])
                
                node = Node(name=port_name, cell_type='InputPort' if direction == 'input' else 'OutputPort')
                node.set_bounding_box(0, 0, 0, 2, 10, 2)
                node.set_pin_location('in', 0, 5, 1)
                node.set_pin_location('out', 2, 5, 1)
                
                if direction == 'input':
                    for bit in bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_outgoing_connection('out', f"net_{bit}")
                else:
                    for bit in bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_incoming_connection('in', f"net_{bit}")
                
                nodes.append(node)

        for cell_name, cell_data in cells.items():
            cell_type = cell_data.get('type', 'unknown')
            node = Node(name=cell_name, cell_type=cell_type)
            
            layout_data = cell_layouts.get(cell_type, {})
            
            if layout_data:
                bbox_data = layout_data.get('bbox', {})
                width = bbox_data.get('width', 0)
                height = bbox_data.get('height', 0)
                depth = bbox_data.get('depth', 0)
                node.set_bounding_box(0, 0, 0, width, height, depth)
                
                for pin_name, pin_pos in layout_data.get('inputs', {}).items():
                    node.set_pin_location(pin_name, pin_pos.get('x', 0), pin_pos.get('y', 0), pin_pos.get('z', 0))
                    
                for pin_name, pin_pos in layout_data.get('outputs', {}).items():
                    node.set_pin_location(pin_name, pin_pos.get('x', 0), pin_pos.get('y', 0), pin_pos.get('z', 0))
            else:
                node.set_bounding_box(0, 0, 0, 2, 2, 2)
                node.set_pin_location('Y', 2, 1, 1)
                node.set_pin_location('A', 0, 1, 1)
                node.set_pin_location('B', 0, 1, 1)
            
            connections = cell_data.get('connections', {})
            port_directions = cell_data.get('port_directions', {})
            
            for port_name, port_bits in connections.items():
                direction = port_directions.get(port_name, None)
                
                is_array = False
                if layout_data:
                    if f"{port_name}[0]" in layout_data.get('inputs', {}) or f"{port_name}[0]" in layout_data.get('outputs', {}):
                        is_array = True
                
                if direction == 'input':
                    for i, bit in enumerate(port_bits):
                        if isinstance(bit, int) and bit >= 0:
                            pin_key = f"{port_name}[{i}]" if is_array else port_name
                            node.add_incoming_connection(pin_key, f"net_{bit}")
                elif direction == 'output':
                    for i, bit in enumerate(port_bits):
                        if isinstance(bit, int) and bit >= 0:
                            pin_key = f"{port_name}[{i}]" if is_array else port_name
                            node.add_outgoing_connection(pin_key, f"net_{bit}")
                else:
                    if port_name in ['A', 'B', 'C', 'D', 'CLK', 'EN', 'ADDR', 'DATA', 'DI', 'WE', 'S']:
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_incoming_connection(pin_key, f"net_{bit}")
                    elif port_name in ['Y', 'Q', 'OUT', 'DO']:
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_outgoing_connection(pin_key, f"net_{bit}")
                    else:
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_incoming_connection(pin_key, f"net_{bit}")
            
            nodes.append(node)
    
    return nodes


def build_graph(nodes: List[Node]) -> nx.DiGraph:
    """Builds a NetworkX graph from the parsed nodes."""
    G = nx.MultiDiGraph()
    
    net_drivers: Dict[str, List[Tuple[str, str]]] = {}
    net_sinks: Dict[str, List[Tuple[str, str]]] = {}
    
    for node in nodes:
        dims = (node.bounding_box[3], node.bounding_box[4], node.bounding_box[5])
        port_count = len(node.incoming) + len(node.outgoing)
        G.add_node(node.name, cell_type=node.cell_type, dims=dims, port_count=port_count, pin_locations=node.pin_locations)
        
        for port, nets in node.outgoing.items():
            for net in nets:
                if net not in net_drivers:
                    net_drivers[net] = []
                net_drivers[net].append((node.name, port))
        
        for port, nets in node.incoming.items():
            for net in nets:
                if net not in net_sinks:
                    net_sinks[net] = []
                net_sinks[net].append((node.name, port))
    
    all_nets = set(net_drivers.keys()) | set(net_sinks.keys())
    for net in all_nets:
        drivers = net_drivers.get(net, [])
        sinks = net_sinks.get(net, [])
        
        for driver_node, driver_port in drivers:
            for sink_node, sink_port in sinks:
                if driver_node != sink_node:
                    G.add_edge(driver_node, sink_node, label=net, weight=5.0, 
                               src_port=driver_port, dst_port=sink_port)
                    
    return G


def optimize_placement(G: nx.DiGraph, max_time_seconds: float = 60.0) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimizes component placement using a force-directed algorithm (Spring Layout)
    followed by a legalization step to resolve collisions and align to grid.
    """
    print("Starting force-directed placement (Spring Layout)...")
    
    input_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'InputPort'])
    output_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'OutputPort'])
    
    n_nodes = len(G.nodes())
    if n_nodes == 0:
        return {}
        
    scale = max(20, int(n_nodes ** (1/3) * 4))
    
    pos = {}
    fixed_nodes = []
    
    if input_ports:
        z_span = scale * 1.0
        z_step = z_span / (len(input_ports) + 1) if len(input_ports) > 0 else 0
        current_z = -z_span / 2 + z_step
        
        for node in input_ports:
            pos[node] = (-scale, 0, current_z)
            current_z += z_step
            
    if output_ports:
        z_span = scale * 1.0
        z_step = z_span / (len(output_ports) + 1) if len(output_ports) > 0 else 0
        current_z = -z_span / 2 + z_step
        
        for node in output_ports:
            pos[node] = (scale, 0, current_z)
            fixed_nodes.append(node)
            current_z += z_step
            
    k_val = scale * 1.5 / (n_nodes ** 0.5) if n_nodes > 0 else 1.0
    
    print(f"Running custom force-directed simulation with Y-penalty (Scale: {scale})...")
    
    pos = nx.spring_layout(
        G, 
        dim=3, 
        k=k_val, 
        pos=pos, 
        fixed=fixed_nodes if fixed_nodes else None, 
        iterations=1000, 
        weight='weight', 
        seed=42, 
        scale=scale
    )
    
    iterations = 100
    t = scale * 0.1
    dt = t / (iterations + 1)
    y_gravity = 0.5
    
    nodes = list(G.nodes())
    node_indices = {n: i for i, n in enumerate(nodes)}
    
    pos_arr = np.array([pos[n] for n in nodes], dtype=float)
    adj = nx.to_scipy_sparse_array(G, weight='weight', nodelist=nodes)
    
    fixed_indices = [node_indices[n] for n in fixed_nodes] if fixed_nodes else []
    fixed_mask = np.zeros(len(nodes), dtype=bool)
    fixed_mask[fixed_indices] = True
    
    for i in range(iterations):
        delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distance, 1.0)
        
        repulsive_strength = (k_val ** 2) / distance
        repulsive_disp = (delta / distance[:, :, np.newaxis]) * repulsive_strength[:, :, np.newaxis]
        disp = np.sum(repulsive_disp, axis=1)
        
        rows, cols = adj.nonzero()
        for u_idx, v_idx in zip(rows, cols):
            if u_idx == v_idx: continue
            
            vec = pos_arr[u_idx] - pos_arr[v_idx]
            dist = np.linalg.norm(vec)
            if dist == 0: continue
            
            snap_strength = 2.0
            snap_force = (k_val * snap_strength) / (dist + 1.0)
            force = (dist ** 2) / k_val + snap_force
            weight = adj[u_idx, v_idx]
            force *= weight
            
            disp_vec = (vec / dist) * force
            disp[u_idx] -= disp_vec
            disp[v_idx] += disp_vec
            
        disp[:, 1] -= pos_arr[:, 1] * y_gravity
        
        length = np.linalg.norm(disp, axis=1)
        length = np.maximum(length, 1e-6)
        scale_factor = np.minimum(length, t) / length
        disp *= scale_factor[:, np.newaxis]
        
        pos_arr[~fixed_mask] += disp[~fixed_mask]
        t -= dt
        
    raw_pos = {n: tuple(pos_arr[i]) for i, n in enumerate(nodes)}
    
    print("Legalizing placement (resolving collisions)...")
    
    node_info = {}
    for node in G.nodes():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
        padding = 6# G.nodes[node].get('padding', h*2) // h
        node_info[node] = {
            'w': w, 'h': h, 'd': d,
            'w_padded': w + padding, 
            'h_padded': h,
            'd_padded': d + padding
        }
        
    final_positions: Dict[str, Tuple[float, float, float]] = {}
    node_rotations: Dict[str, int] = {}  # Store rotation for each node (0, 90, 180, 270)
    occupied_boxes: List[Tuple[int, int, int, int, int, int]] = []
    
    def check_overlap(x: int, y: int, z: int, w: int, h: int, d: int) -> bool:
        for ox, oy, oz, ow, oh, od in occupied_boxes:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return True
        return False
    
    def rotate_dimensions(w: int, h: int, d: int, rotation: int) -> Tuple[int, int, int]:
        """Rotate dimensions. Rotation is 0°, 90°, 180°, 270° around Y axis."""
        if rotation == 0:
            return (w, h, d)
        elif rotation == 90:
            return (d, h, w)  # Swap width and depth
        elif rotation == 180:
            return (w, h, d)  # Same as 0° for bounding box
        else:  # 270
            return (d, h, w)  # Swap width and depth
    
    def rotate_pin_locations(pin_locations: Dict[str, Tuple[int, int, int]], 
                            w: int, h: int, d: int, rotation: int) -> Dict[str, Tuple[int, int, int]]:
        """Rotate pin locations around Y axis."""
        if rotation == 0:
            return pin_locations.copy()
        
        rotated = {}
        for pin_name, (px, py, pz) in pin_locations.items():
            if rotation == 90:
                # Rotate 90° around Y: X -> -Z, Z -> X
                new_x = pz
                new_z = w - px  # w becomes old d
                rotated[pin_name] = (new_x, py, new_z)
            elif rotation == 180:
                # Rotate 180° around Y: X -> -X, Z -> -Z
                new_x = w - px
                new_z = d - pz
                rotated[pin_name] = (new_x, py, new_z)
            else:  # 270
                # Rotate 270° around Y: X -> Z, Z -> -X
                new_x = d - pz  # d becomes old w
                new_z = px
                rotated[pin_name] = (new_x, py, new_z)
        
        return rotated
    
    def calculate_wire_cost(node: str, pos: Tuple[float, float, float], 
                           neighbors: List[str], placed_positions: Dict) -> float:
        """Calculate total Manhattan distance to placed neighbors."""
        cost = 0.0
        for neighbor in neighbors:
            if neighbor in placed_positions:
                nx, ny, nz = placed_positions[neighbor]
                px, py, pz = pos
                cost += abs(nx - px) + abs(ny - py) + abs(nz - pz)
        return cost
        
    def generate_spiral_offsets():
        pq = [(0.0, 0.0, 0, 0, 0)]
        visited = {(0, 0, 0)}
        Y_PENALTY = 10.0
        
        while pq:
            cost, _, dx, dy, dz = heapq.heappop(pq)
            yield (dx, dy, dz)
            
            for nx, ny, nz in [
                (dx+1, dy, dz), (dx-1, dy, dz),
                (dx, dy+1, dz), (dx, dy-1, dz),
                (dx, dy, dz+1), (dx, dy, dz-1)
            ]:
                if (nx, ny, nz) not in visited:
                    visited.add((nx, ny, nz))
                    new_cost = nx*nx + Y_PENALTY * ny*ny + nz*nz
                    heapq.heappush(pq, (new_cost, random.random(), nx, ny, nz))

    sorted_nodes = sorted(G.nodes(), key=lambda n: raw_pos[n][0])
    placed_set = set()
    
    for i, node in enumerate(sorted_nodes):
        if i % 10 == 0:
            print(f"Legalizing {i}/{n_nodes}...", end='\r')
            
        tx, ty, tz = raw_pos[node]
        
        corrections = []
        all_neighbors = list(G.neighbors(node)) + list(G.predecessors(node))
        for neighbor in all_neighbors:
             if neighbor in placed_set:
                 cx, cy, cz = final_positions[neighbor]
                 rx, ry, rz = raw_pos[neighbor]
                 corrections.append((cx-rx, cy-ry, cz-rz))
                 
        if corrections:
            avg_dx = sum(c[0] for c in corrections) / len(corrections)
            avg_dy = sum(c[1] for c in corrections) / len(corrections)
            avg_dz = sum(c[2] for c in corrections) / len(corrections)
            
            damping = 0.8
            tx += avg_dx * damping
            ty += avg_dy * damping
            tz += avg_dz * damping
        
        info = node_info[node]
        base_w, base_h, base_d = info['w_padded'], info['h_padded'], info['d_padded']
        
        # Try all 4 rotations and pick the best one
        best_rotation = 0
        best_cost = float('inf')
        best_position = None
        best_box = None
        
        for rotation in [0, 90, 180, 270]:
            w, h, d = rotate_dimensions(base_w, base_h, base_d, rotation)
            
            start_x = int(tx - w/2)
            start_y = int(ty - h/2)
            start_z = int(tz - d/2)
            
            found_pos = None
            for dx, dy, dz in generate_spiral_offsets():
                x, y, z = start_x + dx, start_y + dy, start_z + dz
                if not check_overlap(x, y, z, w, h, d):
                    found_pos = (x, y, z)
                    break
            
            if found_pos:
                fx, fy, fz = found_pos
                cx = fx + w / 2.0
                cy = fy + h / 2.0
                cz = fz + d / 2.0
                
                # Calculate wire cost for this rotation
                wire_cost = calculate_wire_cost(node, (cx, cy, cz), all_neighbors, final_positions)
                
                # Add penalty for non-zero rotation to prefer 0° when costs are equal
                rotation_penalty = 0.1 * (rotation / 90)
                total_cost = wire_cost + rotation_penalty
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_rotation = rotation
                    best_position = (cx, cy, cz)
                    best_box = (fx, fy, fz, w, h, d)
        
        if best_position is None:
            # Fallback: use 0° rotation if no valid position found
            w, h, d = base_w, base_h, base_d
            start_x = int(tx - w/2)
            start_y = int(ty - h/2)
            start_z = int(tz - d/2)
            for dx, dy, dz in generate_spiral_offsets():
                x, y, z = start_x + dx, start_y + dy, start_z + dz
                if not check_overlap(x, y, z, w, h, d):
                    fx, fy, fz = x, y, z
                    cx = fx + w / 2.0
                    cy = fy + h / 2.0
                    cz = fz + d / 2.0
                    best_position = (cx, cy, cz)
                    best_box = (fx, fy, fz, w, h, d)
                    best_rotation = 0
                    break
        
        occupied_boxes.append(best_box)
        final_positions[node] = best_position
        node_rotations[node] = best_rotation
        placed_set.add(node)
        
        
        # Update node dims in graph to reflect rotation
        original_dims = G.nodes[node].get('dims', (1, 1, 1))
        ow, oh, od = int(original_dims[0]), int(original_dims[1]), int(original_dims[2])
        rotated_dims = rotate_dimensions(ow, oh, od, best_rotation)
        G.nodes[node]['dims'] = rotated_dims
        G.nodes[node]['rotation'] = best_rotation
        
        # Rotate pin locations to match the rotated node
        original_pins = G.nodes[node].get('pin_locations', {})
        rotated_pins = rotate_pin_locations(original_pins, ow, oh, od, best_rotation)
        G.nodes[node]['pin_locations'] = rotated_pins
        
    print(f"\nPlacement complete.")
    return final_positions


class RoutingGrid:
    def __init__(self, positions: Dict[str, Tuple[float, float, float]], nodes_data: Dict[str, dict]):
        self.positions = positions
        self.nodes_data = nodes_data
        self.blocked_coords = set()
        self.node_occupancy = {}
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
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            
            pad = 1
            x_start = int(x - w/2) - pad
            x_end = int(x + w/2) + pad
            y_start = int(y - h/2) - pad
            y_end = int(y + h/2) + pad
            z_start = int(z - d/2) - pad
            z_end = int(z + d/2) + pad
            
            for ix in range(x_start, x_end + 1):
                for iy in range(y_start, y_end + 1):
                    for iz in range(z_start, z_end + 1):
                        coord = (ix, iy, iz)
                        self.blocked_coords.add(coord)
                        self.node_occupancy[coord] = name

    def is_blocked(self, point, allowed_points=None):
        if point in self.blocked_coords:
            if allowed_points and point in allowed_points:
                return False
            return True
        return False

    def add_path(self, path):
        for i in range(len(path)):
            self.blocked_coords.add(path[i])
            
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

    def get_neighbors(self, point, allowed_points=None):
        x, y, z = point
        moves = [
            # Horizontal moves (slope 0) - Cost 1
            ((x+1, y, z), 1.0), ((x-1, y, z), 1.0),
            ((x, y, z+1), 1.0), ((x, y, z-1), 1.0),
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
                
                # Check for clipping on vertical moves
                if abs(ny - y) == 1:
                    dx = nx - x
                    dy = ny - y
                    dz = nz - z
                    mid1 = (x + dx//2, y, z + dz//2)
                    mid2 = (x + dx//2, y + dy, z + dz//2)
                    
                    if self.is_blocked(mid1, allowed_points) or self.is_blocked(mid2, allowed_points):
                        continue

                valid.append(((nx, ny, nz), cost))
        return valid


def a_star(start, goal, grid, allowed_points, max_steps=50000):
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
            
        for neighbor, move_cost in grid.get_neighbors(current, allowed_points):
            if grid.is_blocked(neighbor, allowed_points):
                continue
                
            tentative_g = current_g + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                
    return None


def find_tunnel(grid, start_point):
    """Finds the shortest path from start_point to any non-blocked point."""
    if start_point not in grid.blocked_coords:
        return []
        
    queue = [(start_point, [start_point])]
    visited = {start_point}
    
    while queue:
        current, path = queue.pop(0)
        
        if current not in grid.blocked_coords:
            return path
            
        for neighbor, _ in grid.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    return []

def route_nets(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, List[List[Tuple[int, int, int]]]]:
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
    
    total_nets = len(nets)
    for i, (net_name, net_data) in enumerate(nets.items()):
        if i % 10 == 0:
            print(f"Routing net {i}/{total_nets}...", end='\r')
            
        driver_node, driver_port = net_data['driver']
        sinks = net_data['sinks']
        
        if not driver_node or not sinks:
            continue
            
        d_pos = positions[driver_node]
        d_pins = nodes_data[driver_node]['pin_locations']
        d_pin_offset = d_pins.get(driver_port, (0, 0, 0))
        d_dims = nodes_data[driver_node]['dims']
        
        start_x = int(round(d_pos[0] - d_dims[0]/2 + d_pin_offset[0]))
        start_y = int(round(d_pos[1] - d_dims[1]/2 + d_pin_offset[1]))
        start_z = int(round(d_pos[2] - d_dims[2]/2 + d_pin_offset[2]))
        start_point = (start_x, start_y, start_z)
        
        tree_points = {start_point}
        start_tunnel = find_tunnel(grid, start_point)
        if not start_tunnel and start_point in grid.blocked_coords:
             print(f"  [Fail] Start blocked and no tunnel: {driver_node} {driver_port} at {start_point}")
             continue
        
        remaining_sinks = []
        sink_tunnels = {}
        
        for sink_node, sink_port in sinks:
            s_pos = positions[sink_node]
            s_pins = nodes_data[sink_node]['pin_locations']
            s_pin_offset = s_pins.get(sink_port, (0, 0, 0))
            s_dims = nodes_data[sink_node]['dims']
            
            sx = int(round(s_pos[0] - s_dims[0]/2 + s_pin_offset[0]))
            sy = int(round(s_pos[1] - s_dims[1]/2 + s_pin_offset[1]))
            sz = int(round(s_pos[2] - s_dims[2]/2 + s_pin_offset[2]))
            
            p = (sx, sy, sz)
            remaining_sinks.append(p)
            sink_tunnels[p] = find_tunnel(grid, p)
            if not sink_tunnels[p] and p in grid.blocked_coords:
                print(f"  [Fail] Sink blocked and no tunnel: {sink_node} {sink_port} at {p}")
            
        net_paths = []
        
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
                break
                
            start, end = best_pair
            
            allowed_points = {start, end}
            
            if end in sink_tunnels:
                for p in sink_tunnels[end]:
                    allowed_points.add(p)
            
            for p in start_tunnel:
                allowed_points.add(p)
            for t_path in sink_tunnels.values():
                for p in t_path:
                    allowed_points.add(p)
            
            path = a_star(start, end, grid, allowed_points)
            
            if path:
                net_paths.append(path)
                grid.add_path(path)
                for p in path:
                    tree_points.add(p)
                remaining_sinks.remove(end)
            else:
                print(f"  [Fail] No path found for net {net_name} from {start} to {end}")
                remaining_sinks.remove(end)
                
        routed_paths[net_name] = net_paths
        
    failed_nets = 0
    total_nets = len(nets)
    for net_name, paths in routed_paths.items():
        if not paths and len(nets[net_name]['sinks']) > 0:
            failed_nets += 1
            
    print(f"\nRouting complete. Failed nets: {failed_nets}/{total_nets}")
    return routed_paths

def visualize_graph(G: nx.DiGraph, positions: Optional[Dict[str, Tuple[float, float, float]]] = None, routed_paths: Optional[Dict[str, List[List[Tuple[int, int, int]]]]] = None):
    """Visualizes the graph using a 3D spring layout with approximate bounding boxes."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except ImportError:
        print("Matplotlib and NumPy are required for visualization.")
        return

    shift_vector = np.array([0.0, 0.0, 0.0])

    if positions is None:
        print("Calculating 3D spring layout...")
        pos = nx.spring_layout(G, dim=3, k=10.0, iterations=1000, seed=42)
        scale_factor = 5.0
        for node in pos:
            pos[node] *= scale_factor
    else:
        pos = positions.copy()
        all_coords = np.array(list(pos.values()))
        centroid = np.mean(all_coords, axis=0)
        shift_vector = centroid
        for node in pos:
            pos[node] = (pos[node][0] - centroid[0], 
                         pos[node][1] - centroid[1], 
                         pos[node][2] - centroid[2])

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for node, (x, y, z) in pos.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        
        px = x
        py = z
        pz = y
        
        pw = w
        ph = d
        pd = h
        
        x_min, x_max = px - pw/2, px + pw/2
        y_min, y_max = py - ph/2, py + ph/2
        z_min, z_max = pz - pd/2, pz + pd/2
        
        vertices = [
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min]],
            [[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]],
            [[x_min, y_min, z_min], [x_min, y_max, z_min], [x_min, y_max, z_max], [x_min, y_min, z_max]],
            [[x_max, y_min, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_max, y_min, z_max]],
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max]],
            [[x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max]]
        ]
        
        poly = Poly3DCollection(vertices, alpha=0.3, edgecolor='k')
        
        cell_type = G.nodes[node].get('cell_type', 'unknown')
        color_hash = hash(cell_type) % 0xFFFFFF
        r = ((color_hash >> 16) & 0xFF) / 255.0
        g = ((color_hash >> 8) & 0xFF) / 255.0
        b = (color_hash & 0xFF) / 255.0
        poly.set_facecolor((r, g, b))
        
        ax.add_collection3d(poly)
        ax.text(px, py, pz + pd/2 + 0.5, node, fontsize=8, ha='center')

    if routed_paths:
        print("Drawing routed paths...")
        for net_name, paths in routed_paths.items():
            for path in paths:
                shifted_path = []
                for p in path:
                    sx = p[0] - shift_vector[0]
                    sy = p[1] - shift_vector[1]
                    sz = p[2] - shift_vector[2]
                    shifted_path.append((sx, sy, sz))
                
                xs = [p[0] for p in shifted_path]
                ys = [p[2] for p in shifted_path]
                zs = [p[1] for p in shifted_path]
                
                ax.plot(xs, ys, zs, color='gray', alpha=0.6, linewidth=2)
    else:
        for u, v in G.edges():
            x1, y1, z1 = pos[u]
            x2, y2, z2 = pos[v]
            
            px1, py1, pz1 = x1, z1, y1
            px2, py2, pz2 = x2, z2, y2
            
            ax.plot([px1, px2], [py1, py2], [pz1, pz2], color='gray', alpha=0.5, linewidth=1)

    all_coords = []
    for node, (x, y, z) in pos.items():
        all_coords.append([x, z, y])
    all_coords = np.array(all_coords)
    
    max_dim = 5
    
    if len(all_coords) > 0:
        ax.set_xlim(all_coords[:, 0].min() - max_dim, all_coords[:, 0].max() + max_dim)
        ax.set_ylim(all_coords[:, 1].min() - max_dim, all_coords[:, 1].max() + max_dim)
        ax.set_zlim(all_coords[:, 2].min() - max_dim, all_coords[:, 2].max() + max_dim)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    plt.title("Netlist Graph 3D Layout")
    
    output_file = "netlist_graph.png"
    print(f"Saving graph to {output_file}...")
    plt.savefig(output_file)
    print("Done.")


def visualize_2d_projection(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]], routed_paths: Optional[Dict[str, List[List[Tuple[int, int, int]]]]] = None):
    """Visualizes the graph as a 2D projection on the X-Z plane (top-down)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        print("Matplotlib is required for visualization.")
        return

    print("Generating 2D projection...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for node, (x, y, z) in positions.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        
        px = x - w / 2.0
        py = z - d / 2.0
        
        cell_type = G.nodes[node].get('cell_type', 'unknown')
        color_hash = hash(cell_type) % 0xFFFFFF
        r = ((color_hash >> 16) & 0xFF) / 255.0
        g = ((color_hash >> 8) & 0xFF) / 255.0
        b = (color_hash & 0xFF) / 255.0
        
        rect = Rectangle((px, py), w, d, linewidth=1, edgecolor='black', facecolor=(r, g, b, 0.5))
        ax.add_patch(rect)
        ax.text(x, z, node, fontsize=6, ha='center', va='center', clip_on=True)

    lines = []
    deltas = []
    
    if routed_paths:
        print("Drawing routed paths on 2D projection...")
        for net_name, paths in routed_paths.items():
            for path in paths:
                for i in range(len(path) - 1):
                    p1 = path[i]
                    p2 = path[i+1]
                    
                    pt1 = (p1[0], p1[2])
                    pt2 = (p2[0], p2[2])
                    
                    lines.append([pt1, pt2])
                    
                    avg_y = (p1[1] + p2[1]) / 2.0
                    delta_y = abs(p1[1] - p2[1])
                    metric = avg_y + (10.0 if delta_y > 0 else 0)
                    deltas.append(metric)
    else:
        for u, v in G.edges():
            x1, y1, z1 = positions[u]
            x2, y2, z2 = positions[v]
            
            p1 = (x1, z1)
            p2 = (x2, z2)
            
            lines.append([p1, p2])
            
            delta_y = abs(y1 - y2)
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            metric = delta_y + dist / 2.0
            deltas.append(metric)
        
    if deltas:
        max_delta = max(deltas) if max(deltas) > 0 else 1.0
        norm = plt.Normalize(0, max_delta)
        cmap = plt.get_cmap('plasma')
        
        lc = LineCollection(lines, cmap=cmap, norm=norm, alpha=0.7, linewidths=1.5)
        lc.set_array(deltas)
        ax.add_collection(lc)
        
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Height / Verticality Metric')
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    plt.title("Netlist 2D Projection (X-Z Plane) with Routing")
    
    output_file = "netlist_projection_2d.png"
    print(f"Saving 2D projection to {output_file}...")
    plt.savefig(output_file, dpi=300)
    print("Done.")


def visualize_graph_interactive(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]], routed_paths: Optional[Dict[str, List[List[Tuple[int, int, int]]]]] = None):
    """Visualizes the graph using Plotly for interactive 3D exploration."""
    if go is None:
        print("Plotly is not installed. Skipping interactive visualization.")
        return

    print("Generating interactive 3D visualization...")
    fig = go.Figure()
    
    x_coords = []
    y_coords = []
    z_coords = []
    i_indices = []
    j_indices = []
    k_indices = []
    face_colors = []
    hover_texts = []
    
    def add_box(x, y, z, w, h, d, color_val, name):
        px, py, pz = x, z, y
        pw, ph, pd = w, d, h
        
        x_min, x_max = px - pw/2, px + pw/2
        y_min, y_max = py - ph/2, py + ph/2
        z_min, z_max = pz - pd/2, pz + pd/2
        
        base_idx = len(x_coords)
        
        x_coords.extend([x_min, x_max, x_max, x_min])
        y_coords.extend([y_min, y_min, y_max, y_max])
        z_coords.extend([z_min, z_min, z_min, z_min])
        
        x_coords.extend([x_min, x_max, x_max, x_min])
        y_coords.extend([y_min, y_min, y_max, y_max])
        z_coords.extend([z_max, z_max, z_max, z_max])
        
        i_indices.extend([base_idx+0, base_idx+0])
        j_indices.extend([base_idx+1, base_idx+2])
        k_indices.extend([base_idx+2, base_idx+3])
        
        i_indices.extend([base_idx+4, base_idx+4])
        j_indices.extend([base_idx+5, base_idx+6])
        k_indices.extend([base_idx+6, base_idx+7])
        
        i_indices.extend([base_idx+0, base_idx+0])
        j_indices.extend([base_idx+1, base_idx+5])
        k_indices.extend([base_idx+5, base_idx+4])
        
        i_indices.extend([base_idx+1, base_idx+1])
        j_indices.extend([base_idx+2, base_idx+6])
        k_indices.extend([base_idx+6, base_idx+5])
        
        i_indices.extend([base_idx+2, base_idx+2])
        j_indices.extend([base_idx+3, base_idx+7])
        k_indices.extend([base_idx+7, base_idx+6])
        
        i_indices.extend([base_idx+3, base_idx+3])
        j_indices.extend([base_idx+0, base_idx+4])
        k_indices.extend([base_idx+4, base_idx+7])
        
    cell_types = sorted(list(set(G.nodes[n].get('cell_type', 'unknown') for n in positions)))
    
    import colorsys
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            sat = 0.7 + (random.random() * 0.3)
            val = 0.8 + (random.random() * 0.2)
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
        return colors
        
    type_colors = dict(zip(cell_types, generate_colors(len(cell_types))))
    vertex_colors = []
    node_names = []
    
    for node, (x, y, z) in positions.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        
        cell_type = G.nodes[node].get('cell_type', 'unknown')
        color = type_colors.get(cell_type, 'rgb(200,200,200)')
        
        add_box(x, y, z, w, h, d, 0, node)
        vertex_colors.extend([color] * 8)
        
        info_text = f"{node} ({cell_type})"
        hover_texts.extend([info_text] * 8)
        node_names.append(info_text)

    fig.add_trace(go.Mesh3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        i=i_indices,
        j=j_indices,
        k=k_indices,
        vertexcolor=vertex_colors,
        text=hover_texts,
        opacity=1.0,
        name='Nodes',
        hoverinfo='text',
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.2)
    ))
    
    # Add port indicators (pin locations)
    pin_x = []
    pin_y = []
    pin_z = []
    pin_text = []
    
    for node, (x, y, z) in positions.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        pin_locations = G.nodes[node].get('pin_locations', {})
        
        for pin_name, (pin_ox, pin_oy, pin_oz) in pin_locations.items():
            # Calculate absolute pin position
            abs_x = x - w/2 + pin_ox
            abs_y = y - h/2 + pin_oy
            abs_z = z - d/2 + pin_oz
            
            # Transform to plotly coordinates (swap y and z)
            pin_x.append(abs_x)
            pin_y.append(abs_z)
            pin_z.append(abs_y)
            pin_text.append(f"{node}::{pin_name}")
    
    # Add pin markers
    if pin_x:
        fig.add_trace(go.Scatter3d(
            x=pin_x,
            y=pin_y,
            z=pin_z,
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                symbol='circle',
                line=dict(color='darkred', width=1)
            ),
            text=pin_text,
            hoverinfo='text',
            name='Ports/Pins'
        ))
    

    
    if routed_paths:
        print("Adding routed paths to interactive plot...")
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        for net_name, paths in routed_paths.items():
            for path in paths:
                for i in range(len(path)):
                    p = path[i]
                    edge_x.append(p[0])
                    edge_y.append(p[2])
                    edge_z.append(p[1])
                
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
                
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=5),
            opacity=0.6,
            name='Wires'
        ))

    # Update layout
    fig.update_layout(
        title="Interactive Netlist 3D Layout",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Z (Depth)',
            zaxis_title='Y (Height)',
            aspectmode='data' # Keep aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_file = "netlist_graph.html"
    print(f"Saving interactive graph to {output_file}...")
    fig.write_html(output_file)
    print("Done.")


if __name__ == "__main__":
    import sys
    
    # Default to netlist.json if no argument provided
    netlist_file = sys.argv[1] if len(sys.argv) > 1 else "netlist.json"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    
    print(f"Parsing netlist from {netlist_file}...")
    models = parse_netlist(netlist_file)
    
    print(f"\nFound {len(models)} nodes:\n")
    for model in models:
        print(model)
        
    print("\nBuilding graph...")
    G = build_graph(models)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Try optimization
    optimized_pos = optimize_placement(G, max_time_seconds=timeout)
    
    routed_paths = None
    if optimized_pos:
        print("Using optimized placement.")
        # Run routing
        routed_paths = route_nets(G, optimized_pos)
        visualize_graph(G, positions=optimized_pos, routed_paths=routed_paths)
    else:
        print("Falling back to spring layout.")
        visualize_graph(G)
        
    if optimized_pos:
        visualize_2d_projection(G, optimized_pos, routed_paths=routed_paths)
        visualize_graph_interactive(G, optimized_pos, routed_paths=routed_paths)


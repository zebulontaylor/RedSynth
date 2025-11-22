#!/usr/bin/env python3
"""
Netlist parser for Yosys JSON output.
"""
import json
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import heapq
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
        # Bounding box: (x, y, z, width, height, depth)
        self.bounding_box: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
        # Incoming connections: dict of {port_name: [connection_keys]}
        self.incoming: Dict[str, List[str]] = {}
        # Outgoing connections: dict of {port_name: [connection_keys]}
        self.outgoing: Dict[str, List[str]] = {}
        # Pin locations: dict of {port_name: (x, y, z)} relative to node origin
        self.pin_locations: Dict[str, Tuple[int, int, int]] = {}
    
    def set_bounding_box(self, x: int, y: int, z: int, width: int, height: int, depth: int):
        """Set the bounding box for this node."""
        self.bounding_box = (x, y, z, width, height, depth)
    
    def add_incoming_connection(self, port: str, connection_key: str):
        """Add an incoming connection."""
        if port not in self.incoming:
            self.incoming[port] = []
        self.incoming[port].append(connection_key)
    
    def add_outgoing_connection(self, port: str, connection_key: str):
        """Add an outgoing connection."""
        if port not in self.outgoing:
            self.outgoing[port] = []
        self.outgoing[port].append(connection_key)
        
    def set_pin_location(self, port: str, x: int, y: int, z: int):
        """Set the relative location of a pin."""
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
    
    # Iterate through all modules in the netlist
    for module_name, module_data in netlist.get('modules', {}).items():
        # Process cells in this module
        cells = module_data.get('cells', {})
        ports = module_data.get('ports', {})
        
        # Only process the top module or modules with cells
        # Primitives usually don't have cells (or are blackboxes)
        if not cells and not ports:
            continue

        # Process Ports (Inputs/Outputs)
        # We only want to do this for the top-level module to avoid creating ports for primitives
        # The top module usually has the 'top' attribute set to 1
        attributes = module_data.get('attributes', {})
        is_top = attributes.get('top', 0) == 1 or str(attributes.get('top', '0')).strip() == '1' or str(attributes.get('top', '0')).strip().endswith('1')
        
        # If we can't determine top, we heuristic: if it has cells, it's likely a module we want to render
        if cells:
            for port_name, port_data in ports.items():
                direction = port_data.get('direction', 'input')
                bits = port_data.get('bits', [])
                
                # Create a node for the port
                node = Node(name=port_name, cell_type='InputPort' if direction == 'input' else 'OutputPort')
                
                # Set a default size for ports (tall and thin)
                node.set_bounding_box(0, 0, 0, 2, 10, 2) 
                
                # Set default pin location (center-ish)
                node.set_pin_location('in', 1, 5, 1)
                node.set_pin_location('out', 1, 5, 1)
                
                if direction == 'input':
                    # Input port drives nets (Outgoing)
                    for bit in bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_outgoing_connection('out', f"net_{bit}")
                else:
                    # Output port receives nets (Incoming)
                    for bit in bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_incoming_connection('in', f"net_{bit}")
                
                nodes.append(node)

        # Process cells in this module
        for cell_name, cell_data in cells.items():
            cell_type = cell_data.get('type', 'unknown')
            node = Node(name=cell_name, cell_type=cell_type)
            
            layout_data = cell_layouts.get(cell_type, {})
            
            # Set bounding box from cell layout if available
            if layout_data:
                bbox_data = layout_data.get('bbox', {})
                width = bbox_data.get('width', 0)
                height = bbox_data.get('height', 0)
                depth = bbox_data.get('depth', 0)
                # Set position to (0,0,0) for now, and use the dimensions from cells.json
                node.set_bounding_box(0, 0, 0, width, height, depth)
                
                # Set pin locations
                inputs = layout_data.get('inputs', {})
                outputs = layout_data.get('outputs', {})
                
                for pin_name, pin_pos in inputs.items():
                    node.set_pin_location(pin_name, pin_pos.get('x', 0), pin_pos.get('y', 0), pin_pos.get('z', 0))
                    
                for pin_name, pin_pos in outputs.items():
                    node.set_pin_location(pin_name, pin_pos.get('x', 0), pin_pos.get('y', 0), pin_pos.get('z', 0))
                    
            else:
                # Default size
                node.set_bounding_box(0, 0, 0, 2, 2, 2)
                # Default pin
                node.set_pin_location('Y', 1, 1, 1)
                node.set_pin_location('A', 1, 1, 1)
                node.set_pin_location('B', 1, 1, 1)
            
            # Process connections
            connections = cell_data.get('connections', {})
            port_directions = cell_data.get('port_directions', {})
            
            for port_name, port_bits in connections.items():
                # Use port_directions if available, otherwise fall back to heuristic
                direction = port_directions.get(port_name, None)
                
                # Handle array ports (e.g., A[0], A[1])
                # If the port name in connections is 'A' but layout has 'A[0]', 'A[1]'...
                # We need to map the bits to the specific pins.
                
                # Check if this port is an array in layout
                is_array = False
                if layout_data:
                    # Simple check: if 'A[0]' exists in inputs/outputs
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
                    # Fallback heuristic
                    if port_name in ['A', 'B', 'C', 'D', 'CLK', 'EN', 'ADDR', 'DATA', 'DI', 'WE', 'S']:
                        # Input port
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_incoming_connection(pin_key, f"net_{bit}")
                    elif port_name in ['Y', 'Q', 'OUT', 'DO']:
                        # Output port
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_outgoing_connection(pin_key, f"net_{bit}")
                    else:
                        # Unknown, treat as input
                        for i, bit in enumerate(port_bits):
                            if isinstance(bit, int) and bit >= 0:
                                pin_key = f"{port_name}[{i}]" if is_array else port_name
                                node.add_incoming_connection(pin_key, f"net_{bit}")
            
            nodes.append(node)
    
    return nodes


def build_graph(nodes: List[Node]) -> nx.DiGraph:
    """Builds a NetworkX graph from the parsed nodes."""
    G = nx.DiGraph()
    
    # Map net names to producers and consumers
    net_drivers: Dict[str, List[Tuple[str, str]]] = {} # net -> [(node_name, port_name)]
    net_sinks: Dict[str, List[Tuple[str, str]]] = {}   # net -> [(node_name, port_name)]
    
    for node in nodes:
        # Store bbox dimensions in the graph node
        # bbox is (x, y, z, width, height, depth)
        # We only care about dimensions (width, height, depth) for visualization shape
        dims = (node.bounding_box[3], node.bounding_box[4], node.bounding_box[5])
        # Store port count for padding calculation
        port_count = len(node.incoming) + len(node.outgoing)
        G.add_node(node.name, cell_type=node.cell_type, dims=dims, port_count=port_count, pin_locations=node.pin_locations)
        
        # Process outgoing (drivers)
        for port, nets in node.outgoing.items():
            for net in nets:
                if net not in net_drivers:
                    net_drivers[net] = []
                net_drivers[net].append((node.name, port))
        
        # Process incoming (sinks)
        for port, nets in node.incoming.items():
            for net in nets:
                if net not in net_sinks:
                    net_sinks[net] = []
                net_sinks[net].append((node.name, port))
    
    # Create edges
    all_nets = set(net_drivers.keys()) | set(net_sinks.keys())
    for net in all_nets:
        drivers = net_drivers.get(net, [])
        sinks = net_sinks.get(net, [])
        
        for driver_node, driver_port in drivers:
            for sink_node, sink_port in sinks:
                if driver_node != sink_node:
                    # Store port names in edge data for routing
                    G.add_edge(driver_node, sink_node, label=net, weight=5.0, 
                               src_port=driver_port, dst_port=sink_port)
                    
    return G


def optimize_placement(G: nx.DiGraph, max_time_seconds: float = 60.0) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimizes component placement using a force-directed algorithm (Spring Layout)
    followed by a legalization step to resolve collisions and align to grid.
    """
    print("Starting force-directed placement (Spring Layout)...")
    
    # Identify Input and Output ports for fixed placement
    input_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'InputPort'])
    output_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'OutputPort'])
    
    n_nodes = len(G.nodes())
    if n_nodes == 0:
        return {}
        
    # Reduced scale for tighter packing
    # Heuristic: approximate side length of a cube containing all nodes
    # Previous was * 8, reducing to * 4 for even tighter pack
    scale = max(20, int(n_nodes ** (1/3) * 4))
    
    # Setup fixed positions for ports
    pos = {}
    fixed_nodes = []
    
    # Place Inputs on the Left (Negative X)
    if input_ports:
        # Spread along Z axis (vertical in 2D projection)
        z_span = scale * 1.0
        z_step = z_span / (len(input_ports) + 1) if len(input_ports) > 0 else 0
        current_z = -z_span / 2 + z_step
        
        for node in input_ports:
            pos[node] = (-scale, 0, current_z)
            # fixed_nodes.append(node) # Allow inputs to move
            current_z += z_step
            
    # Place Outputs on the Right (Positive X)
    if output_ports:
        # Spread along Z axis
        z_span = scale * 1.0
        z_step = z_span / (len(output_ports) + 1) if len(output_ports) > 0 else 0
        current_z = -z_span / 2 + z_step
        
        for node in output_ports:
            pos[node] = (scale, 0, current_z)
            fixed_nodes.append(node)
            current_z += z_step
            
    # Run Spring Layout
    # k is the optimal distance between nodes.
    # We want it small relative to scale to encourage clustering.
    # Default is 1/sqrt(n). We scale it up by our scale.
    k_val = scale * 1.5 / (n_nodes ** 0.5) if n_nodes > 0 else 1.0
    
    # Custom Force-Directed Layout with Y-Penalty
    # We start with the spring layout initialization but run our own loop
    # to inject a gravity force on the Y axis.
    
    print(f"Running custom force-directed simulation with Y-penalty (Scale: {scale})...")
    
    # Initial positions using standard spring layout for a good starting point
    # Run for fewer iterations to get a rough shape
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
    
    # Custom Simulation Loop
    # Parameters
    iterations = 100
    t = scale * 0.1 # Initial temperature
    dt = t / (iterations + 1)
    
    # Y-Penalty factor: Force pulling towards Y=0
    y_gravity = 0.5 
    
    nodes = list(G.nodes())
    node_indices = {n: i for i, n in enumerate(nodes)}
    
    # Convert pos to numpy array for faster processing
    pos_arr = np.array([pos[n] for n in nodes], dtype=float)
    
    # Adjacency matrix (sparse)
    adj = nx.to_scipy_sparse_array(G, weight='weight', nodelist=nodes)
    
    # Fixed node indices
    fixed_indices = [node_indices[n] for n in fixed_nodes] if fixed_nodes else []
    fixed_mask = np.zeros(len(nodes), dtype=bool)
    fixed_mask[fixed_indices] = True
    
    for i in range(iterations):
        # 1. Repulsive forces (between all pairs)
        # Simplified: just random sampling or use a grid if too slow, 
        # but for < 1000 nodes, N^2 is fine-ish.
        
        delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        
        # Avoid division by zero
        np.fill_diagonal(distance, 1.0) 
        
        # Repulsive force: k^2 / d
        # Direction: delta / distance
        repulsive_strength = (k_val ** 2) / distance
        repulsive_disp = (delta / distance[:, :, np.newaxis]) * repulsive_strength[:, :, np.newaxis]
        
        # Sum repulsive forces
        disp = np.sum(repulsive_disp, axis=1)
        
        # 2. Attractive forces (connected pairs)
        # For each edge
        rows, cols = adj.nonzero()
        for u_idx, v_idx in zip(rows, cols):
            if u_idx == v_idx: continue
            
            vec = pos_arr[u_idx] - pos_arr[v_idx]
            dist = np.linalg.norm(vec)
            if dist == 0: continue
            
            # Attractive force: d^2 / k
            # Modified to include a short-range "snap" force to prioritize short connections
            # F = d^2/k + C/(d+1)
            # The 1/d term makes the potential concave at short range, favoring 
            # "1 short + 1 long" over "2 medium" connections.
            
            snap_strength = 2.0
            snap_force = (k_val * snap_strength) / (dist + 1.0)
            
            force = (dist ** 2) / k_val + snap_force
            
            # Apply weight
            weight = adj[u_idx, v_idx]
            force *= weight
            
            disp_vec = (vec / dist) * force
            
            # Pull u towards v, v towards u
            disp[u_idx] -= disp_vec
            disp[v_idx] += disp_vec
            
        # 3. Y-Penalty (Gravity towards Y=0)
        # Pull nodes towards Y=0 based on their current Y height
        # Force = -y * gravity
        disp[:, 1] -= pos_arr[:, 1] * y_gravity
        
        # 4. Update positions
        # Limit displacement by temperature
        length = np.linalg.norm(disp, axis=1)
        length = np.maximum(length, 1e-6) # Avoid div zero
        
        # Cap displacement at temperature t
        scale_factor = np.minimum(length, t) / length
        disp *= scale_factor[:, np.newaxis]
        
        # Apply displacement (skip fixed nodes)
        pos_arr[~fixed_mask] += disp[~fixed_mask]
        
        # Cool down
        t -= dt
        
    # Update raw_pos with new positions
    raw_pos = {n: tuple(pos_arr[i]) for i, n in enumerate(nodes)}
    
    print("Legalizing placement (resolving collisions)...")
    
    # --- Legalization / Collision Resolution ---
    
    # Pre-calculate node dimensions and padding
    node_info = {}
    for node in G.nodes():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
        padding = G.nodes[node].get('padding', h*2) // h
        node_info[node] = {
            'w': w, 'h': h, 'd': d,
            'w_padded': w + padding, 
            'h_padded': h, # No padding on Y (stacking)
            'd_padded': d + padding
        }
        
    final_positions: Dict[str, Tuple[float, float, float]] = {}
    occupied_boxes: List[Tuple[int, int, int, int, int, int]] = [] # (x, y, z, w, h, d)
    
    def check_overlap(x: int, y: int, z: int, w: int, h: int, d: int) -> bool:
        for ox, oy, oz, ow, oh, od in occupied_boxes:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return True
        return False
        
    # Generator for spiral offsets (reused from before)
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

    # Process nodes: Fixed ports first, then others sorted by X coordinate (sweep line-ish)
    # This helps keep the flow from left to right
    sorted_nodes = sorted(G.nodes(), key=lambda n: raw_pos[n][0])
    
    placed_set = set()
    
    for i, node in enumerate(sorted_nodes):
        if i % 10 == 0:
            print(f"Legalizing {i}/{n_nodes}...", end='\r')
            
        # Target position from spring layout
        tx, ty, tz = raw_pos[node]
        
        # Displacement Propagation:
        # Adjust target based on how neighbors were shifted during their legalization.
        # This helps preserve local clusters (short connections).
        # Target = Raw(u) + Average(Final(v) - Raw(v)) for placed v
        
        corrections = []
        # Check both incoming and outgoing neighbors
        all_neighbors = list(G.neighbors(node)) + list(G.predecessors(node))
        for neighbor in all_neighbors:
             if neighbor in placed_set:
                 cx, cy, cz = final_positions[neighbor]
                 rx, ry, rz = raw_pos[neighbor]
                 corrections.append((cx-rx, cy-ry, cz-rz))
                 
        if corrections:
            # Average correction
            avg_dx = sum(c[0] for c in corrections) / len(corrections)
            avg_dy = sum(c[1] for c in corrections) / len(corrections)
            avg_dz = sum(c[2] for c in corrections) / len(corrections)
            
            # Apply correction with damping
            damping = 0.8
            tx += avg_dx * damping
            ty += avg_dy * damping
            tz += avg_dz * damping
        
        info = node_info[node]
        w, h, d = info['w_padded'], info['h_padded'], info['d_padded']
        
        # Snap to grid (center -> top-left)
        start_x = int(tx - w/2)
        start_y = int(ty - h/2)
        start_z = int(tz - d/2)
        
        # Find nearest valid spot
        found_pos = None
        
        for dx, dy, dz in generate_spiral_offsets():
            x, y, z = start_x + dx, start_y + dy, start_z + dz
            
            if not check_overlap(x, y, z, w, h, d):
                found_pos = (x, y, z)
                break
                
        # Record
        fx, fy, fz = found_pos
        occupied_boxes.append((fx, fy, fz, w, h, d))
        
        # Convert back to center for result
        cx = fx + w / 2.0
        cy = fy + h / 2.0
        cz = fz + d / 2.0
        final_positions[node] = (cx, cy, cz)
        placed_set.add(node)
        
    print(f"\nPlacement complete.")
    return final_positions


class RoutingGrid:
    def __init__(self, positions: Dict[str, Tuple[float, float, float]], nodes_data: Dict[str, dict]):
        self.positions = positions
        self.nodes_data = nodes_data
        # Use a set for blocked coordinates for O(1) lookup
        self.blocked_coords = set()
        self.node_occupancy = {} # (x,y,z) -> node_name
        self.min_coords = [float('inf')] * 3
        self.max_coords = [float('-inf')] * 3
        self._build_grid()

    def _build_grid(self):
        # Determine bounds
        for pos in self.positions.values():
            for i in range(3):
                self.min_coords[i] = min(self.min_coords[i], pos[i])
                self.max_coords[i] = max(self.max_coords[i], pos[i])
        
        # Add padding to bounds
        padding = 10
        self.min_coords = [int(x - padding) for x in self.min_coords]
        self.max_coords = [int(x + padding) for x in self.max_coords]

        print("Building routing grid obstacles...")
        # Mark obstacles
        # Optimization: Pre-calculate integer bounds for all nodes
        for name, pos in self.positions.items():
            dims = self.nodes_data[name].get('dims', (1, 1, 1))
            w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            
            # Node occupies [x-w/2, x+w/2] etc.
            # We inflate by 1 for wire width
            pad = 0
            
            x_start = int(x - w/2) - pad
            x_end = int(x + w/2) + pad
            y_start = int(y - h/2) - pad
            y_end = int(y + h/2) + pad
            z_start = int(z - d/2) - pad
            z_end = int(z + d/2) + pad
            
            # Optimization: Iterate ranges directly
            # This is still potentially slow if nodes are huge, but better than before
            for ix in range(x_start, x_end + 1):
                for iy in range(y_start, y_end + 1):
                    for iz in range(z_start, z_end + 1):
                        coord = (ix, iy, iz)
                        self.blocked_coords.add(coord)
                        # We store which node blocks this coordinate to allow start/end access
                        # If multiple nodes block, we just overwrite (it's still blocked)
                        self.node_occupancy[coord] = name

    def is_blocked(self, point, allowed_points):
        if point in self.blocked_coords:
            # If the point is explicitly allowed (it's a start or end pin), it's NOT blocked
            if point in allowed_points:
                return False
            return True
        return False

    def get_neighbors(self, point):
        x, y, z = point
        # 6-connectivity
        moves = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        # Filter out-of-bounds
        # Optimization: Inline checks
        valid = []
        min_x, min_y, min_z = self.min_coords
        max_x, max_y, max_z = self.max_coords
        
        for nx, ny, nz in moves:
            if (min_x <= nx <= max_x and
                min_y <= ny <= max_y and
                min_z <= nz <= max_z):
                valid.append((nx, ny, nz))
        return valid


def a_star(start, goal, grid, allowed_points, max_steps=10000):
    # Priority queue: (f_score, g_score, current_node)
    # Optimization: Use Manhattan distance as heuristic
    
    start_node = start
    goal_node = goal
    
    # Heuristic function
    def h(p1, p2):
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
            # Fail if taking too long
            return None
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        for neighbor in grid.get_neighbors(current):
            if grid.is_blocked(neighbor, allowed_points):
                continue
                
            tentative_g = current_g + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                
    return None


def find_tunnel(grid, start_point):
    """Finds the shortest path from start_point to any non-blocked point."""
    # If start point is already free (except for being in blocked_coords), we are good.
    # But blocked_coords includes the node itself.
    # We want to find a point that is NOT in blocked_coords.
    
    if start_point not in grid.blocked_coords:
        return []
        
    queue = [(start_point, [start_point])]
    visited = {start_point}
    
    # BFS
    while queue:
        current, path = queue.pop(0)
        
        if current not in grid.blocked_coords:
            return path
            
        for neighbor in grid.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    return [] # Should not happen if grid is bounded and padded

def route_nets(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, List[List[Tuple[int, int, int]]]]:
    print("Starting routing...")
    
    nodes_data = {}
    for node in G.nodes():
        nodes_data[node] = {
            'dims': G.nodes[node].get('dims', (1, 1, 1)),
            'pin_locations': G.nodes[node].get('pin_locations', {})
        }
        
    grid = RoutingGrid(positions, nodes_data)
    
    # Group edges by net
    nets = {}
    for u, v, data in G.edges(data=True):
        net_name = data.get('label', 'unknown')
        src_port = data.get('src_port')
        dst_port = data.get('dst_port')
        
        if net_name not in nets:
            nets[net_name] = {'driver': None, 'sinks': []}
        
        # Assuming one driver per net for now (or we pick one)
        # We store the specific port used for connection
        if not nets[net_name]['driver']:
             nets[net_name]['driver'] = (u, src_port)
        
        nets[net_name]['sinks'].append((v, dst_port))
        
    routed_paths = {}
    
    total_nets = len(nets)
    for i, (net_name, net_data) in enumerate(nets.items()):
        if i % 10 == 0:
            print(f"Routing net {i}/{total_nets}...", end='\r')
            
        driver_node, driver_port = net_data['driver']
        sinks = net_data['sinks'] # List of (node, port)
        
        if not driver_node or not sinks:
            continue
            
        # Allowed points for this net: Start point + All Sink points
        # We need to allow the specific pin locations to be traversed (start/end)
        # The rest of the node volume remains blocked.
        
        # Calculate absolute start position based on pin location
        d_pos = positions[driver_node]
        d_pins = nodes_data[driver_node]['pin_locations']
        # Default to center if pin not found
        d_pin_offset = d_pins.get(driver_port, (0, 0, 0))
        
        d_dims = nodes_data[driver_node]['dims']
        start_x = int(d_pos[0] - d_dims[0]/2 + d_pin_offset[0])
        start_y = int(d_pos[1] - d_dims[1]/2 + d_pin_offset[1])
        start_z = int(d_pos[2] - d_dims[2]/2 + d_pin_offset[2])
        start_point = (start_x, start_y, start_z)
        
        tree_points = {start_point}
        
        # Pre-calculate tunnels for start point
        start_tunnel = find_tunnel(grid, start_point)
        
        remaining_sinks = []
        sink_tunnels = {} # point -> tunnel path
        
        for sink_node, sink_port in sinks:
            s_pos = positions[sink_node]
            s_pins = nodes_data[sink_node]['pin_locations']
            s_pin_offset = s_pins.get(sink_port, (0, 0, 0))
            s_dims = nodes_data[sink_node]['dims']
            
            sx = int(s_pos[0] - s_dims[0]/2 + s_pin_offset[0])
            sy = int(s_pos[1] - s_dims[1]/2 + s_pin_offset[1])
            sz = int(s_pos[2] - s_dims[2]/2 + s_pin_offset[2])
            
            p = (sx, sy, sz)
            remaining_sinks.append(p)
            sink_tunnels[p] = find_tunnel(grid, p)
            
        net_paths = []
        
        while remaining_sinks:
            # Optimization: Instead of checking ALL tree points, 
            # just check the last added path points or a sample?
            # Or just check start_point and endpoints of existing paths?
            # For now, let's stick to checking all but maybe optimize the loop.
            
            best_dist = float('inf')
            best_pair = None
            
            # Heuristic optimization:
            # Just find the closest sink to the INITIAL start point first?
            # No, that defeats the purpose of a Steiner tree.
            # But checking 1000 tree points vs 10 sinks is 10k checks. Fast enough in Python?
            # Maybe.
            
            # Limit tree_points check to a subset if it gets too large?
            # Or use a KD-tree? Overkill.
            
            # Let's just loop.
            for sp in remaining_sinks:
                for tp in tree_points:
                     dist = abs(tp[0]-sp[0]) + abs(tp[1]-sp[1]) + abs(tp[2]-sp[2])
                     if dist < best_dist:
                         best_dist = dist
                         best_pair = (tp, sp)
            
            if not best_pair:
                break
                
            start, end = best_pair
            
            # Allowed points: 
            # 1. The start point (on tree)
            # 2. The specific end point (sink)
            # 3. The tunnel from start (if start was a pin or on a tunnel) - wait, tree_points includes tunnels now?
            # 4. The tunnel for the end point
            
            allowed_points = {start, end}
            
            # Add tunnel points for end
            if end in sink_tunnels:
                for p in sink_tunnels[end]:
                    allowed_points.add(p)
                    
            # Add tunnel points for start? 
            # 'start' is in tree_points. 
            # If 'start' is the original driver pin, we need its tunnel.
            # If 'start' is somewhere else on the routed path, it should be free (or already tunneled).
            # But if we are branching off a point that was part of a tunnel, we need to ensure that tunnel is still allowed.
            # Actually, once a point is in tree_points, it means we routed to it.
            # If we routed to it, it must have been allowed.
            # But if we are starting a NEW search from it, we need to pass it as allowed again.
            # Yes, 'start' is in allowed_points.
            
            # But if 'start' is deep inside a node (because it's a pin), we need to allow the path OUT of it.
            # If we already routed TO it, the path OUT is the same path? No.
            # We need to allow the tunnel *associated* with 'start' if 'start' is a pin.
            
            # Simplification: Just add ALL tunnels for this net to allowed_points?
            # That might allow routing through other sinks' tunnels?
            # Probably fine, as they are tunnels to free space.
            # But strictly, we only need the tunnel for 'end' and the tunnel for 'start' (if it's a pin).
            
            # Let's just add start_tunnel and sink_tunnels[end]
            for p in start_tunnel:
                allowed_points.add(p)
            
            # Also, if 'start' happens to be one of the other sinks we already routed to?
            # We should cache tunnels for all points in tree?
            # Or just add ALL tunnels for the net. It's safer and easier.
            for p in start_tunnel:
                allowed_points.add(p)
            for t_path in sink_tunnels.values():
                for p in t_path:
                    allowed_points.add(p)
            
            # Run A*
            path = a_star(start, end, grid, allowed_points)
            
            if path:
                net_paths.append(path)
                # Add path to tree
                for p in path:
                    tree_points.add(p)
                remaining_sinks.remove(end)
            else:
                # If failed, maybe try routing from original start?
                # Or just skip
                remaining_sinks.remove(end)
                
        routed_paths[net_name] = net_paths
        
    # Count failures
    failed_nets = 0
    total_nets = len(nets)
    for net_name, paths in routed_paths.items():
        # A net is failed if it has sinks but no paths (or fewer paths than sinks)
        # Actually, we just check if paths is empty but sinks existed
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
        # dim=3 for 3D layout
        pos = nx.spring_layout(G, dim=3, k=10.0, iterations=1000, seed=42)
        
        # Scale the positions to give some room for the boxes
        # Reduced scale factor as requested
        scale_factor = 5.0
        for node in pos:
            pos[node] *= scale_factor
    else:
        pos = positions.copy()
        # Center the positions around 0,0,0 for better viewing
        all_coords = np.array(list(pos.values()))
        centroid = np.mean(all_coords, axis=0)
        shift_vector = centroid
        for node in pos:
            pos[node] = (pos[node][0] - centroid[0], 
                         pos[node][1] - centroid[1], 
                         pos[node][2] - centroid[2])

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw nodes as boxes
    for node, (x, y, z) in pos.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        
        # Swap Y and Z for visualization (Y is up in model, Z is up in plot)
        # Model: (x, y, z) -> size (w, h, d)
        # Plot:  (x', y', z') where x'=x, y'=z, z'=y
        # Size:  (w', h', d') where w'=w, h'=d, d'=h
        
        # Calculate corners of the box centered at (x, z, y)
        # x, y, z are the center coordinates from the model
        
        # Plot X = Model X
        # Plot Y = Model Z
        # Plot Z = Model Y
        
        px = x
        py = z
        pz = y
        
        # Dimensions in plot coordinates
        pw = w
        ph = d # Plot Y dimension is Model Z dimension (depth)
        pd = h # Plot Z dimension is Model Y dimension (height)
        
        x_min, x_max = px - pw/2, px + pw/2
        y_min, y_max = py - ph/2, py + ph/2
        z_min, z_max = pz - pd/2, pz + pd/2
        
        # Define vertices for the 6 faces
        vertices = [
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min]], # Bottom
            [[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]], # Top
            [[x_min, y_min, z_min], [x_min, y_max, z_min], [x_min, y_max, z_max], [x_min, y_min, z_max]], # Left
            [[x_max, y_min, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_max, y_min, z_max]], # Right
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max]], # Front
            [[x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max]]  # Back
        ]
        
        # Create 3D box
        poly = Poly3DCollection(vertices, alpha=0.3, edgecolor='k')
        
        # Color based on cell type (simple hash)
        cell_type = G.nodes[node].get('cell_type', 'unknown')
        color_hash = hash(cell_type) % 0xFFFFFF
        r = ((color_hash >> 16) & 0xFF) / 255.0
        g = ((color_hash >> 8) & 0xFF) / 255.0
        b = (color_hash & 0xFF) / 255.0
        poly.set_facecolor((r, g, b))
        
        ax.add_collection3d(poly)
        
        # Add label slightly above the box
        ax.text(px, py, pz + pd/2 + 0.5, node, fontsize=8, ha='center')

    # Draw edges
    if routed_paths:
        print("Drawing routed paths...")
        for net_name, paths in routed_paths.items():
            for path in paths:
                # Shift path points
                shifted_path = []
                for p in path:
                    sx = p[0] - shift_vector[0]
                    sy = p[1] - shift_vector[1]
                    sz = p[2] - shift_vector[2]
                    shifted_path.append((sx, sy, sz))
                
                # Plot: X -> X, Y -> Z, Z -> Y
                xs = [p[0] for p in shifted_path]
                ys = [p[2] for p in shifted_path]
                zs = [p[1] for p in shifted_path]
                
                ax.plot(xs, ys, zs, color='gray', alpha=0.6, linewidth=2)
    else:
        for u, v in G.edges():
            x1, y1, z1 = pos[u]
            x2, y2, z2 = pos[v]
            
            # Apply same transform
            px1, py1, pz1 = x1, z1, y1
            px2, py2, pz2 = x2, z2, y2
            
            ax.plot([px1, px2], [py1, py2], [pz1, pz2], color='gray', alpha=0.5, linewidth=1)

    # Set axis limits to show everything
    # We need to recalculate limits based on transformed coordinates
    all_coords = []
    for node, (x, y, z) in pos.items():
        all_coords.append([x, z, y]) # Transformed
    all_coords = np.array(all_coords)
    
    # Add some padding based on max dimensions
    max_dim = 5 # heuristic
    
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
    """
    Visualizes the graph as a 2D projection on the X-Z plane (top-down).
    Wires are colored based on vertical (Y) span.
    """
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
    
    # Draw nodes as rectangles
    # Model X -> Plot X
    # Model Z -> Plot Y (Depth)
    
    for node, (x, y, z) in positions.items():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        
        # Calculate top-left corner for Rectangle (x, y)
        # Position is center, so subtract half dimensions
        # Plot X = Model X
        # Plot Y = Model Z
        
        px = x - w / 2.0
        py = z - d / 2.0
        
        # Color based on cell type
        cell_type = G.nodes[node].get('cell_type', 'unknown')
        color_hash = hash(cell_type) % 0xFFFFFF
        r = ((color_hash >> 16) & 0xFF) / 255.0
        g = ((color_hash >> 8) & 0xFF) / 255.0
        b = (color_hash & 0xFF) / 255.0
        
        rect = Rectangle((px, py), w, d, linewidth=1, edgecolor='black', facecolor=(r, g, b, 0.5))
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, z, node, fontsize=6, ha='center', va='center', clip_on=True)

    # Prepare edges for LineCollection
    lines = []
    deltas = []
    
    if routed_paths:
        print("Drawing routed paths on 2D projection...")
        for net_name, paths in routed_paths.items():
            for path in paths:
                # Path is list of (x, y, z)
                # We want to draw segments
                for i in range(len(path) - 1):
                    p1 = path[i]
                    p2 = path[i+1]
                    
                    # Project to X-Z
                    # Model X -> Plot X
                    # Model Z -> Plot Y
                    
                    pt1 = (p1[0], p1[2])
                    pt2 = (p2[0], p2[2])
                    
                    lines.append([pt1, pt2])
                    
                    # Metric: Average Y height + Delta Y
                    # This helps visualize which layer the wire is on
                    avg_y = (p1[1] + p2[1]) / 2.0
                    delta_y = abs(p1[1] - p2[1])
                    
                    # If vertical segment (delta_y > 0), highlight it
                    metric = avg_y + (10.0 if delta_y > 0 else 0)
                    deltas.append(metric)
    else:
        # Draw straight lines
        for u, v in G.edges():
            x1, y1, z1 = positions[u]
            x2, y2, z2 = positions[v]
            
            # Project to X-Z
            p1 = (x1, z1)
            p2 = (x2, z2)
            
            lines.append([p1, p2])
            
            delta_y = abs(y1 - y2)
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            
            # Metric: Vertical Span + Distance / Constant
            metric = delta_y + dist / 2.0
            deltas.append(metric)
        
    # Normalize deltas for colormap
    if deltas:
        max_delta = max(deltas) if max(deltas) > 0 else 1.0
        norm = plt.Normalize(0, max_delta)
        cmap = plt.get_cmap('plasma') # plasma is good for intensity
        
        lc = LineCollection(lines, cmap=cmap, norm=norm, alpha=0.7, linewidths=1.5)
        lc.set_array(deltas)
        ax.add_collection(lc)
        
        # Add colorbar
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
    """
    Visualizes the graph using Plotly for interactive 3D exploration.
    """
    if go is None:
        print("Plotly is not installed. Skipping interactive visualization.")
        return

    print("Generating interactive 3D visualization...")
    
    # Create figure
    fig = go.Figure()
    
    # Add nodes as 3D boxes
    # Plotly Mesh3d is good for arbitrary shapes, but for simple boxes we can use a trick
    # or just use scatter3d for vertices and mesh3d for faces.
    # A simpler way for many boxes is to use a single Mesh3d with all vertices and faces combined,
    # or use 'go.Mesh3d' for each box (might be slow for many boxes).
    # Let's try one Mesh3d for all boxes to be efficient? 
    # No, we want different colors/hover info per node.
    # So we'll use one trace per node group or just one trace per node if N is small (<1000).
    # Given 700 nodes, one trace per node might be heavy but manageable. 
    # Better: Use a single Scatter3d for all centers with custom markers? No, we want boxes.
    # Best for performance with boxes: Use a single Mesh3d for all boxes.
    
    # Let's construct arrays for a single Mesh3d
    x_coords = []
    y_coords = []
    z_coords = []
    i_indices = []
    j_indices = []
    k_indices = []
    
    # Colors
    face_colors = []
    hover_texts = []
    
    # Helper to add a box
    def add_box(x, y, z, w, h, d, color_val, name):
        # Vertices
        # Plot X = Model X
        # Plot Y = Model Z
        # Plot Z = Model Y
        
        px, py, pz = x, z, y
        pw, ph, pd = w, d, h
        
        x_min, x_max = px - pw/2, px + pw/2
        y_min, y_max = py - ph/2, py + ph/2
        z_min, z_max = pz - pd/2, pz + pd/2
        
        # 8 vertices
        base_idx = len(x_coords)
        
        # Bottom face (z_min)
        x_coords.extend([x_min, x_max, x_max, x_min])
        y_coords.extend([y_min, y_min, y_max, y_max])
        z_coords.extend([z_min, z_min, z_min, z_min])
        
        # Top face (z_max)
        x_coords.extend([x_min, x_max, x_max, x_min])
        y_coords.extend([y_min, y_min, y_max, y_max])
        z_coords.extend([z_max, z_max, z_max, z_max])
        
        # 12 triangles (2 per face * 6 faces)
        # Indices are relative to base_idx
        
        # Bottom: 0,1,2; 0,2,3
        i_indices.extend([base_idx+0, base_idx+0])
        j_indices.extend([base_idx+1, base_idx+2])
        k_indices.extend([base_idx+2, base_idx+3])
        
        # Top: 4,5,6; 4,6,7
        i_indices.extend([base_idx+4, base_idx+4])
        j_indices.extend([base_idx+5, base_idx+6])
        k_indices.extend([base_idx+6, base_idx+7])
        
        # Front: 0,1,5; 0,5,4
        i_indices.extend([base_idx+0, base_idx+0])
        j_indices.extend([base_idx+1, base_idx+5])
        k_indices.extend([base_idx+5, base_idx+4])
        
        # Right: 1,2,6; 1,6,5
        i_indices.extend([base_idx+1, base_idx+1])
        j_indices.extend([base_idx+2, base_idx+6])
        k_indices.extend([base_idx+6, base_idx+5])
        
        # Back: 2,3,7; 2,7,6
        i_indices.extend([base_idx+2, base_idx+2])
        j_indices.extend([base_idx+3, base_idx+7])
        k_indices.extend([base_idx+7, base_idx+6])
        
        # Left: 3,0,4; 3,4,7
        i_indices.extend([base_idx+3, base_idx+3])
        j_indices.extend([base_idx+0, base_idx+4])
        k_indices.extend([base_idx+4, base_idx+7])
        
        # Color
        # We need one color per vertex? Or per face?
        # Mesh3d supports vertexcolor or facecolor.
        # Let's use intensity.
        
    # We will use 'vertexcolor' to color nodes distinctly by type.
    # Generate colors for each cell type
    cell_types = sorted(list(set(G.nodes[n].get('cell_type', 'unknown') for n in positions)))
    
    import colorsys
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0 # Golden ratio
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
        
        # Add box vertices
        add_box(x, y, z, w, h, d, 0, node) # color_val unused now
        
        # Add color for 8 vertices
        vertex_colors.extend([color] * 8)
        
        # For hover, we can't easily do per-face hover in a single mesh.
        # We'll add a separate Scatter3d for hover points (centers)
        node_names.append(f"{node} ({cell_type})")

    # Add the Mesh3d trace for all boxes
    fig.add_trace(go.Mesh3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        i=i_indices,
        j=j_indices,
        k=k_indices,
        vertexcolor=vertex_colors, # Use explicit colors
        opacity=1.0,
        name='Nodes',
        hoverinfo='skip', # We use scatter for hover
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.2) # Better lighting
    ))
    
    # Add Scatter3d for hover info
    # Plot X = Model X
    # Plot Y = Model Z
    # Plot Z = Model Y
    scatter_x = [positions[n][0] for n in positions]
    scatter_y = [positions[n][2] for n in positions]
    scatter_z = [positions[n][1] for n in positions]
    
    fig.add_trace(go.Scatter3d(
        x=scatter_x,
        y=scatter_y,
        z=scatter_z,
        mode='markers',
        marker=dict(size=2, color='rgba(0,0,0,0)'), # Invisible markers
        text=node_names,
        hoverinfo='text',
        name='Node Info'
    ))
    
    # Draw routed paths
    if routed_paths:
        print("Adding routed paths to interactive plot...")
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        for net_name, paths in routed_paths.items():
            for path in paths:
                for i in range(len(path)):
                    p = path[i]
                    # Plot X = Model X
                    # Plot Y = Model Z
                    # Plot Z = Model Y
                    edge_x.append(p[0])
                    edge_y.append(p[2])
                    edge_z.append(p[1])
                
                # Add None to break line between paths
                edge_x.append(None)
                edge_y.append(None)
                edge_z.append(None)
                
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=5), # Bolder wires (width=5) and black color
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


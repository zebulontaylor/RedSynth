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
            
            # Set bounding box from cell layout if available
            if cell_type in cell_layouts:
                bbox_data = cell_layouts[cell_type].get('bbox', {})
                width = bbox_data.get('width', 0)
                height = bbox_data.get('height', 0)
                depth = bbox_data.get('depth', 0)
                # Set position to (0,0,0) for now, and use the dimensions from cells.json
                node.set_bounding_box(0, 0, 0, width, height, depth)
            else:
                # Default size
                node.set_bounding_box(0, 0, 0, 2, 2, 2)
            
            # Process connections
            connections = cell_data.get('connections', {})
            port_directions = cell_data.get('port_directions', {})
            
            for port_name, port_bits in connections.items():
                # Use port_directions if available, otherwise fall back to heuristic
                direction = port_directions.get(port_name, None)
                
                if direction == 'input':
                    for bit in port_bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_incoming_connection(port_name, f"net_{bit}")
                elif direction == 'output':
                    for bit in port_bits:
                        if isinstance(bit, int) and bit >= 0:
                            node.add_outgoing_connection(port_name, f"net_{bit}")
                else:
                    # Fallback heuristic
                    if port_name in ['A', 'B', 'C', 'D', 'CLK', 'EN', 'ADDR', 'DATA', 'DI', 'WE', 'S']:
                        # Input port
                        for bit in port_bits:
                            if isinstance(bit, int) and bit >= 0:
                                node.add_incoming_connection(port_name, f"net_{bit}")
                    elif port_name in ['Y', 'Q', 'OUT', 'DO']:
                        # Output port
                        for bit in port_bits:
                            if isinstance(bit, int) and bit >= 0:
                                node.add_outgoing_connection(port_name, f"net_{bit}")
                    else:
                        # Unknown, treat as input
                        for bit in port_bits:
                            if isinstance(bit, int) and bit >= 0:
                                node.add_incoming_connection(port_name, f"net_{bit}")
            
            nodes.append(node)
    
    return nodes


def build_graph(nodes: List[Node]) -> nx.DiGraph:
    """Builds a NetworkX graph from the parsed nodes."""
    G = nx.DiGraph()
    
    # Map net names to producers and consumers
    net_drivers: Dict[str, List[str]] = {}
    net_sinks: Dict[str, List[str]] = {}
    
    for node in nodes:
        # Store bbox dimensions in the graph node
        # bbox is (x, y, z, width, height, depth)
        # We only care about dimensions (width, height, depth) for visualization shape
        dims = (node.bounding_box[3], node.bounding_box[4], node.bounding_box[5])
        # Store port count for padding calculation
        port_count = len(node.incoming) + len(node.outgoing)
        G.add_node(node.name, cell_type=node.cell_type, dims=dims, port_count=port_count)
        
        # Process outgoing (drivers)
        for port, nets in node.outgoing.items():
            for net in nets:
                if net not in net_drivers:
                    net_drivers[net] = []
                net_drivers[net].append(node.name)
        
        # Process incoming (sinks)
        for port, nets in node.incoming.items():
            for net in nets:
                if net not in net_sinks:
                    net_sinks[net] = []
                net_sinks[net].append(node.name)
    
    # Create edges
    all_nets = set(net_drivers.keys()) | set(net_sinks.keys())
    for net in all_nets:
        drivers = net_drivers.get(net, [])
        sinks = net_sinks.get(net, [])
        
        for driver in drivers:
            for sink in sinks:
                if driver != sink:
                    G.add_edge(driver, sink, label=net)
                    
    return G


def optimize_placement(G: nx.DiGraph, max_time_seconds: float = 60.0) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimizes component placement using a constructive grid-based spiral search.
    Prioritizes placing connected components close to each other.
    Penalizes height (Y) differences more than horizontal (X, Z) differences.
    """
    print("Starting grid-based spiral search placement...")
    
    # Configuration
    Y_PENALTY = 10.0  # Cost multiplier for Y (height) differences (squared). Increased to flatten layout.
    
    # Pre-calculate node dimensions and padding
    node_info = {}
    for node in G.nodes():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        
        w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
        # Calculate padding: fixed small amount to ensure separation but high density
        padding = 2
        
        # Apply padding to X and Z, but not Y (stacking allowed/encouraged)
        w_padded = w + padding
        d_padded = d + padding
        h_padded = h
        
        node_info[node] = {
            'w': w, 'h': h, 'd': d,
            'w_padded': w_padded, 'h_padded': h_padded, 'd_padded': d_padded
        }
        
    placed_positions: Dict[str, Tuple[int, int, int]] = {} # Top-left-front corner
    occupied_boxes: List[Tuple[int, int, int, int, int, int]] = [] # (x, y, z, w, h, d)
    
    # Helper to check overlap
    def check_overlap(x: int, y: int, z: int, w: int, h: int, d: int) -> bool:
        for ox, oy, oz, ow, oh, od in occupied_boxes:
            # Check for intersection in all 3 dimensions
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return True
        return False

    # --- Pre-place Input and Output Ports ---
    input_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'InputPort'])
    output_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'OutputPort'])
    
    # Estimate grid size
    total_nodes = len(G.nodes())
    estimated_side = int(total_nodes ** 0.5) * 6 # Increased multiplier to allow more 2D space
    
    # Place Inputs on the Left (Negative X)
    # Bring them closer to the center
    input_x = -estimated_side // 2 - 5 
    
    # Calculate total height of inputs to center them
    total_input_height = 0
    for node in input_ports:
        info = node_info[node]
        total_input_height += info['d_padded'] + 1 # +1 for spacing
        
    current_z = -total_input_height // 2
    
    for node in input_ports:
        info = node_info[node]
        w, h, d = info['w_padded'], info['h_padded'], info['d_padded']
        
        # Place
        x, y, z = input_x, 0, current_z
        placed_positions[node] = (x, y, z)
        occupied_boxes.append((x, y, z, w, h, d))
        
        current_z += d + 1 # Tighter spacing
        
    # Place Outputs on the Right (Positive X)
    output_x = estimated_side // 2 + 5
    
    total_output_height = 0
    for node in output_ports:
        info = node_info[node]
        total_output_height += info['d_padded'] + 1
        
    current_z = -total_output_height // 2
    
    for node in output_ports:
        info = node_info[node]
        w, h, d = info['w_padded'], info['h_padded'], info['d_padded']
        
        # Place
        x, y, z = output_x, 0, current_z
        placed_positions[node] = (x, y, z)
        occupied_boxes.append((x, y, z, w, h, d))
        
        current_z += d + 1

    # Generator for spiral offsets
    def generate_spiral_offsets():
        """Yields (dx, dy, dz) sorted by weighted distance."""
        # Priority queue stores (weighted_cost, random_tie_breaker, dx, dy, dz)
        # Cost = dx^2 + Y_PENALTY * dy^2 + dz^2
        # Random tie-breaker ensures isotropic growth (avoids axis bias)
        pq = [(0.0, 0.0, 0, 0, 0)]
        visited = {(0, 0, 0)}
        
        while pq:
            cost, _, dx, dy, dz = heapq.heappop(pq)
            yield (dx, dy, dz)
            
            # Expand neighbors
            for nx, ny, nz in [
                (dx+1, dy, dz), (dx-1, dy, dz),
                (dx, dy+1, dz), (dx, dy-1, dz),
                (dx, dy, dz+1), (dx, dy, dz-1)
            ]:
                if (nx, ny, nz) not in visited:
                    visited.add((nx, ny, nz))
                    new_cost = nx*nx + Y_PENALTY * ny*ny + nz*nz
                    heapq.heappush(pq, (new_cost, random.random(), nx, ny, nz))

    # Determine placement order: BFS/Connectivity based
    degrees = dict(G.degree())
    if not degrees:
        return {}
        
    # We will maintain a set of unplaced nodes
    unplaced = set(G.nodes()) - set(placed_positions.keys())
    
    # Main placement loop
    while unplaced:
        # Pick next node to place
        # Prefer nodes connected to already placed nodes
        best_candidate = None
        
        # If we have placed nodes, look for neighbors of placed nodes
        candidates = []
        if placed_positions:
            for node in unplaced:
                # Count connections to placed nodes
                connections = 0
                for neighbor in G.neighbors(node):
                    if neighbor in placed_positions:
                        connections += 1
                for predecessor in G.predecessors(node):
                    if predecessor in placed_positions:
                        connections += 1
                
                if connections > 0:
                    candidates.append((connections, degrees[node], node))
            
            # Sort by connections (desc), then total degree (desc)
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        if candidates:
            node_to_place = candidates[0][2]
        else:
            # Disconnected component or start, pick max degree from unplaced
            node_to_place = max(unplaced, key=lambda n: degrees[n])
            
        unplaced.remove(node_to_place)
        info = node_info[node_to_place]
        w, h, d = info['w_padded'], info['h_padded'], info['d_padded']
        
        # Calculate ideal position (centroid of placed neighbors)
        neighbor_positions = []
        for neighbor in list(G.neighbors(node_to_place)) + list(G.predecessors(node_to_place)):
            if neighbor in placed_positions:
                nx_pos, ny_pos, nz_pos = placed_positions[neighbor]
                n_info = node_info[neighbor]
                # Use center of neighbor for centroid calculation
                neighbor_positions.append((
                    nx_pos + n_info['w_padded']/2,
                    ny_pos + n_info['h_padded']/2,
                    nz_pos + n_info['d_padded']/2
                ))
        
        if neighbor_positions:
            # Median is robust for Manhattan distance
            avg_x = np.median([p[0] for p in neighbor_positions])
            avg_y = np.median([p[1] for p in neighbor_positions])
            avg_z = np.median([p[2] for p in neighbor_positions])
            
            # Align to grid (roughly center the new node on the ideal point)
            start_x = int(avg_x - w/2)
            start_y = int(avg_y - h/2)
            start_z = int(avg_z - d/2)
        else:
            start_x, start_y, start_z = 0, 0, 0
            
        # Spiral search for valid spot
        found_pos = None
        found_dims = None
        
        search_limit = 1000000 
        count = 0
        
        for dx, dy, dz in generate_spiral_offsets():
            x = start_x + dx
            y = start_y + dy
            z = start_z + dz
            
            # Try orientations
            valid_orientations = []
            
            # Orientation 1: Original
            if not check_overlap(x, y, z, w, h, d):
                # Calculate cost (Manhattan distance to neighbors)
                cx = x + w / 2.0
                cy = y + h / 2.0
                cz = z + d / 2.0
                cost = 0
                for nx_pos, ny_pos, nz_pos in neighbor_positions:
                    cost += abs(cx - nx_pos) + abs(cy - ny_pos) + abs(cz - nz_pos)
                valid_orientations.append((cost, (w, h, d)))
            
            # Orientation 2: Rotated 90 deg
            if w != d:
                if not check_overlap(x, y, z, d, h, w):
                    cx = x + d / 2.0
                    cy = y + h / 2.0
                    cz = z + w / 2.0
                    cost = 0
                    for nx_pos, ny_pos, nz_pos in neighbor_positions:
                        cost += abs(cx - nx_pos) + abs(cy - ny_pos) + abs(cz - nz_pos)
                    valid_orientations.append((cost, (d, h, w)))
            
            if valid_orientations:
                # Pick best orientation (min cost)
                valid_orientations.sort(key=lambda x: x[0])
                found_pos = (x, y, z)
                found_dims = valid_orientations[0][1]
                break
            
            count += 1
            if count > search_limit:
                print(f"Warning: Search limit reached for node {node_to_place}")
                found_pos = (x, y, z)
                found_dims = (w, h, d)
                break
                
        placed_positions[node_to_place] = found_pos
        
        # Update node info with chosen dimensions
        fw, fh, fd = found_dims
        node_info[node_to_place]['w_padded'] = fw
        node_info[node_to_place]['h_padded'] = fh
        node_info[node_to_place]['d_padded'] = fd
        
        # Update original dims (subtract padding)
        padding = 2
        node_info[node_to_place]['w'] = fw - padding
        node_info[node_to_place]['h'] = fh
        node_info[node_to_place]['d'] = fd - padding
        
        # Update graph dims for visualization
        G.nodes[node_to_place]['dims'] = (fw - padding, fh, fd - padding)
        
        occupied_boxes.append((found_pos[0], found_pos[1], found_pos[2], fw, fh, fd))
        
        if len(placed_positions) % 10 == 0:
            print(f"Placed {len(placed_positions)}/{len(G.nodes())} nodes...", end='\r')
            
    print(f"\nPlacement complete.")
    
    # Convert to centers for visualization return
    final_positions = {}
    for node, (x, y, z) in placed_positions.items():
        info = node_info[node]
        cx = x + info['w_padded'] / 2.0
        cy = y + info['h_padded'] / 2.0
        cz = z + info['d_padded'] / 2.0
        final_positions[node] = (cx, cy, cz)
        
    return final_positions



def visualize_graph(G: nx.DiGraph, positions: Optional[Dict[str, Tuple[float, float, float]]] = None):
    """Visualizes the graph using a 3D spring layout with approximate bounding boxes."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except ImportError:
        print("Matplotlib and NumPy are required for visualization.")
        return

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
        pos = positions
        # Center the positions around 0,0,0 for better viewing
        all_coords = np.array(list(pos.values()))
        centroid = np.mean(all_coords, axis=0)
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


def visualize_2d_projection(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]]):
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
    colors = []
    
    # We want to color by Delta Y
    deltas = []
    
    for u, v in G.edges():
        x1, y1, z1 = positions[u]
        x2, y2, z2 = positions[v]
        
        # Project to X-Z
        p1 = (x1, z1)
        p2 = (x2, z2)
        
        lines.append([p1, p2])
        
        delta_y = abs(y1 - y2)
        deltas.append(delta_y)
        
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
        cbar.set_label('Vertical Span (Delta Y)')
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    plt.title("Netlist 2D Projection (X-Z Plane)")
    
    output_file = "netlist_projection_2d.png"
    print(f"Saving 2D projection to {output_file}...")
    plt.savefig(output_file, dpi=300)
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
    
    if optimized_pos:
        print("Using optimized placement.")
        visualize_graph(G, positions=optimized_pos)
    else:
        print("Falling back to spring layout.")
        visualize_graph(G)
        
    if optimized_pos:
        visualize_2d_projection(G, optimized_pos)


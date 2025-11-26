from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import heapq
import random
import math

# Global cache for spiral offsets to avoid recomputation
_SPIRAL_OFFSETS_CACHE = None

class SpatialIndex:
    """
    A simple spatial hashing index for fast 3D collision detection.
    Divides space into buckets of fixed size.
    """
    def __init__(self, bucket_size=10):
        self.bucket_size = bucket_size
        self.buckets: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int, int, int]]] = {}

    def _get_bucket_keys(self, x, y, z, w, h, d):
        """Get all bucket keys that a box overlaps with."""
        min_bx = int(x // self.bucket_size)
        max_bx = int((x + w) // self.bucket_size)
        min_by = int(y // self.bucket_size)
        max_by = int((y + h) // self.bucket_size)
        min_bz = int(z // self.bucket_size)
        max_bz = int((z + d) // self.bucket_size)

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for bz in range(min_bz, max_bz + 1):
                    yield (bx, by, bz)

    def insert(self, box: Tuple[int, int, int, int, int, int]):
        x, y, z, w, h, d = box
        buckets = self.buckets
        bucket_append = list.append
        for key in self._get_bucket_keys(x, y, z, w, h, d):
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = []
            bucket_append(bucket, box)

    def query(self, x, y, z, w, h, d) -> bool:
        """Returns True if the box overlaps with any existing box in the index."""
        buckets = self.buckets

        min_bx = int(x // self.bucket_size)
        max_bx = int((x + w) // self.bucket_size)
        min_by = int(y // self.bucket_size)
        max_by = int((y + h) // self.bucket_size)
        min_bz = int(z // self.bucket_size)
        max_bz = int((z + d) // self.bucket_size)

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for bz in range(min_bz, max_bz + 1):
                    bucket = buckets.get((bx, by, bz))
                    if not bucket:
                        continue
                    for ox, oy, oz, ow, oh, od in bucket:
                        # Fast AABB check
                        if (x < ox + ow and x + w > ox and
                            y < oy + oh and y + h > oy and
                            z < oz + od and z + d > oz):
                            return True
        return False


def _generate_spiral_offsets(max_offsets: int = 250_000, step: int = 2) -> List[Tuple[int, int, int]]:
    """Generate spiral offsets in increasing cost order and cache the result."""
    offsets: List[Tuple[int, int, int]] = []
    pq: List[Tuple[float, float, int, int, int]] = [(0.0, 0.0, 0, 0, 0)]
    visited = {(0, 0, 0)}
    rand = random.Random(42)
    y_penalty = 40.0

    while pq and len(offsets) < max_offsets:
        cost, _, dx, dy, dz = heapq.heappop(pq)
        offsets.append((dx, dy, dz))

        for next_x, next_y, next_z in (
            (dx + step, dy, dz), (dx - step, dy, dz),
            (dx, dy + step, dz), (dx, dy - step, dz),
            (dx, dy, dz + step), (dx, dy, dz - step),
        ):
            key = (next_x, next_y, next_z)
            if key in visited:
                continue
            visited.add(key)
            effective_y_penalty = y_penalty * abs(next_y) / (abs(next_x) + abs(next_z) + 1)
            effective_y_penalty = max(effective_y_penalty, 2.0)
            new_cost = next_x * next_x + effective_y_penalty * next_y * next_y + next_z * next_z
            heapq.heappush(pq, (new_cost, rand.random(), next_x, next_y, next_z))

    return offsets


def _get_spiral_offsets() -> List[Tuple[int, int, int]]:
    global _SPIRAL_OFFSETS_CACHE
    if _SPIRAL_OFFSETS_CACHE is None:
        print("Generating spiral offsets cache...")
        _SPIRAL_OFFSETS_CACHE = _generate_spiral_offsets()
    return _SPIRAL_OFFSETS_CACHE


def calculate_spring_layout(G: nx.DiGraph, max_time_seconds: float = 60.0) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculates the initial placement using a force-directed algorithm (Spring Layout).
    Returns the raw positions before legalization.
    """
    print("Starting force-directed placement (Spring Layout)...")
    
    input_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'InputPort'])
    output_ports = sorted([n for n in G.nodes() if G.nodes[n].get('cell_type') == 'OutputPort'])
    
    n_nodes = len(G.nodes())
    if n_nodes == 0:
        return {}
        
    scale = max(10, int(n_nodes ** (1/2) * 9))
    
    pos = {}
    fixed_nodes = []
    
    if input_ports:
        z_span = scale * 1.0
        z_step = z_span / (len(input_ports) + 1) if len(input_ports) > 0 else 0
        current_z = -z_span / 2 + z_step
        
        for node in input_ports:
            pos[node] = (-scale, 0, current_z)
            fixed_nodes.append(node)
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
        pos=pos if pos else None, 
        fixed=fixed_nodes if fixed_nodes else None, 
        iterations=2000,
        weight='weight', 
        seed=42,
        scale=scale
    )
    
    iterations = 1000
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

    # Identify port indices for repulsion
    port_nodes = set(input_ports + output_ports)
    port_indices = [node_indices[n] for n in port_nodes]
    port_mask = np.zeros(len(nodes), dtype=bool)
    port_mask[port_indices] = True
    
    PORT_PADDING = 20.0
    PORT_REPULSION_STRENGTH = 5.0
    
    for i in range(iterations):
        delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distance, 1.0)
        
        repulsive_strength = (k_val ** 2) / distance
        
        # Apply extra repulsion from ports
        # We want to push non-ports away from ports
        # Create a mask where rows are ports and cols are non-ports (or vice versa)
        # But simpler: just increase repulsive strength if one of the nodes is a port
        
        # Broadcasting port_mask to shape (N, N)
        is_port_row = port_mask[:, np.newaxis] # Shape (N, 1)
        is_port_col = port_mask[np.newaxis, :] # Shape (1, N)
        
        # If either is a port (but not both? actually both is fine, ports are fixed anyway or should repel each other)
        # We specifically care about Port <-> Non-Port interaction
        # But general port repulsion is fine too.
        
        port_interaction = is_port_row | is_port_col
        
        # Increase repulsion for port interactions
        repulsive_strength[port_interaction] *= PORT_REPULSION_STRENGTH
        
        # Add "hard" padding force
        # If distance < PORT_PADDING and one is a port, apply massive force
        close_to_port = port_interaction & (distance < PORT_PADDING) & (distance > 0)
        repulsive_strength[close_to_port] *= 10.0
        
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
            disp_vec[1] *= 5.0 # Increase vertical stiffness to keep connected nodes at similar heights
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
    return raw_pos


def legalize_placement(G: nx.DiGraph, raw_pos: Dict[str, Tuple[float, float, float]], 
                       padding_phases: List[int] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    Legalizes the placement by resolving collisions and aligning to grid.
    Uses multiple phases with decreasing padding to allow repositioning of nodes.
    
    Args:
        G: The netlist graph
        raw_pos: Initial positions from spring layout
        padding_phases: List of padding values to use in each phase (decreasing order)
                       Default: [24, 12]
    """
    if padding_phases is None:
        padding_phases = [24, 16]
    
    print(f"Legalizing placement with {len(padding_phases)} phases (padding: {padding_phases})...")
    
    n_nodes = len(G.nodes())
    
    # Store original node properties (dims, pin_locations) to restore between phases
    original_node_props = {}
    for node in G.nodes():
        original_node_props[node] = {
            'dims': G.nodes[node].get('dims', (1, 1, 1)),
            'pin_locations': G.nodes[node].get('pin_locations', {}).copy()
        }
    
    final_positions: Dict[str, Tuple[float, float, float]] = {}
    node_rotations: Dict[str, int] = {}  # Store rotation for each node (0, 90, 180, 270)
    
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
                           neighbors: List[str], placed_positions: Dict, 
                           raw_positions: Dict) -> float:
        """Calculate total Manhattan distance to neighbors (placed or raw)."""
        cost = 0.0
        for neighbor in neighbors:
            if neighbor in placed_positions:
                n_x, n_y, n_z = placed_positions[neighbor]
                px, py, pz = pos
                cost += abs(n_x - px) + abs(n_y - py) + abs(n_z - pz)
            elif neighbor in raw_positions:
                # Fallback to raw spring layout position if not yet placed
                n_x, n_y, n_z = raw_positions[neighbor]
                px, py, pz = raw_positions[node]
                # Dont weight wire cost for raw positions as heavily
                cost += (abs(n_x - px) + abs(n_y - py) + abs(n_z - pz)) * 0.5
            else:
                continue
                
        return cost
    
    def calculate_total_wire_cost(positions: Dict[str, Tuple[float, float, float]]) -> float:
        """Calculate total wire cost for all edges in the graph."""
        total_cost = 0.0
        for u, v in G.edges():
            if u in positions and v in positions:
                ux, uy, uz = positions[u]
                vx, vy, vz = positions[v]
                total_cost += abs(ux - vx) + abs(uy - vy) + abs(uz - vz)
        return total_cost
        
    # Get cached spiral offsets
    spiral_offsets_cache = _get_spiral_offsets()

    # Run multiple phases with decreasing padding
    current_pos = raw_pos.copy()  # Track legalized positions for neighbor corrections
    
    # Print initial wire cost from raw spring layout
    initial_wire_cost = calculate_total_wire_cost(raw_pos)
    print(f"Initial wire cost (from spring layout): {initial_wire_cost:.1f}")
    
    for phase_idx, padding in enumerate(padding_phases):
        print(f"\n=== Phase {phase_idx + 1}/{len(padding_phases)} (padding={padding}) ===")
        
        # Restore original node properties before each phase
        for node in G.nodes():
            G.nodes[node]['dims'] = original_node_props[node]['dims']
            G.nodes[node]['pin_locations'] = original_node_props[node]['pin_locations'].copy()
        
        # Calculate node_info with current padding
        node_info = {}
        for node in G.nodes():
            dims = G.nodes[node].get('dims', (1, 1, 1))
            w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
            current_padding = padding
            current_padding = current_padding + (current_padding % 2)  # Ensure padding is even
            node_info[node] = {
                'w': w, 'h': h, 'd': d,
                'w_padded': w + current_padding, 
                'h_padded': h + 2,  # make sure no pins conflict vertically
                'd_padded': d + current_padding
            }
        
        # Reset spatial index for this phase
        spatial_index = SpatialIndex(bucket_size=20)
        
        # Reset phase-local tracking
        phase_positions: Dict[str, Tuple[float, float, float]] = {}
        phase_rotations: Dict[str, int] = {}
        placed_set = set()
        
        # Vary processing order between phases to give different nodes priority
        # Phase 0: left-to-right (by X), Phase 1: right-to-left, Phase 2: shuffle, etc.
        all_nodes = list(G.nodes())
        if phase_idx % 4 == 0:
            sorted_nodes = sorted(all_nodes, key=lambda n: raw_pos[n][0])  # Left to right
        elif phase_idx % 4 == 1:
            sorted_nodes = sorted(all_nodes, key=lambda n: -raw_pos[n][0])  # Right to left
        elif phase_idx % 4 == 2:
            sorted_nodes = sorted(all_nodes, key=lambda n: raw_pos[n][2])  # Front to back (Z)
        else:
            sorted_nodes = sorted(all_nodes, key=lambda n: -raw_pos[n][2])  # Back to front
        
        for i, node in enumerate(sorted_nodes):
            if i % 10 == 0:
                print(f"  Placing {i}/{n_nodes}...", end='\r')
            
            # Always use raw spring layout position as base target (ideal relative positioning)
            tx, ty, tz = raw_pos[node]
            
            # Apply corrections based on where neighbors actually got placed vs. their raw positions
            corrections = []
            all_neighbors = list(G.neighbors(node)) + list(G.predecessors(node))
            for neighbor in all_neighbors:
                 if neighbor in placed_set:
                     cx, cy, cz = phase_positions[neighbor]
                     rx, ry, rz = raw_pos[neighbor]
                     corrections.append((cx-rx, cy-ry, cz-rz))
                     
            if corrections:
                avg_dx = sum(c[0] for c in corrections) / len(corrections)
                avg_dy = sum(c[1] for c in corrections) / len(corrections)
                avg_dz = sum(c[2] for c in corrections) / len(corrections)
                
                # Increase damping in later phases for more stability
                damping = 0.9 - (phase_idx * 0.1)  # 0.9, 0.8, 0.7, 0.6...
                damping = max(damping, 0.5)
                tx += avg_dx * damping
                ty += avg_dy * damping
                tz += avg_dz * damping
            
            info = node_info[node]
            base_w, base_h, base_d = info['w_padded'], info['h_padded'], info['d_padded']
            
            # Try rotations: only test all rotations in later phases or if 0° fails
            # In first phase, just try 0° for speed
            best_rotation = 0
            best_cost = float('inf')
            best_position = None
            best_box = None
            
            rotations_to_try = [0] if phase_idx == 0 else [0, 90, 270]
            
            for rotation in rotations_to_try:
                w, h, d = rotate_dimensions(base_w, base_h, base_d, rotation)
                
                # Snap to even coordinates
                start_x = int(tx - w/2)
                start_x = start_x + (start_x % 2)
                
                start_y = int(ty - h/2)
                start_y = start_y + (start_y % 2)
                
                start_z = int(tz - d/2)
                start_z = start_z + (start_z % 2)
                
                found_pos = None
                for dx, dy, dz in spiral_offsets_cache:
                    x, y, z = start_x + dx, start_y + dy, start_z + dz
                    if not spatial_index.query(x, y, z, w, h, d):
                        found_pos = (x, y, z)
                        break
                
                if found_pos:
                    fx, fy, fz = found_pos
                    cx = fx + w / 2.0
                    cy = fy + h / 2.0
                    cz = fz + d / 2.0
                    
                    # Calculate wire cost for this rotation
                    wire_cost = calculate_wire_cost(node, (cx, cy, cz), all_neighbors, phase_positions, raw_pos)
                    
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
                # Snap to even coordinates
                start_x = int(tx - w/2)
                start_x = start_x + (start_x % 2)
                
                start_y = int(ty - h/2)
                start_y = start_y + (start_y % 2)
                
                start_z = int(tz - d/2)
                start_z = start_z + (start_z % 2)
                
                # If cache was exhausted, we might need to search further or just stack it far away
                # For now, just use the last valid spot found or force place (shouldn't happen often with large cache)
                print(f"\nWarning: Could not find non-overlapping position for {node} within search limit.")
                fx, fy, fz = start_x, start_y, start_z
                cx = fx + w / 2.0
                cy = fy + h / 2.0
                cz = fz + d / 2.0
                best_position = (cx, cy, cz)
                best_box = (fx, fy, fz, w, h, d)
                best_rotation = 0
            
            spatial_index.insert(best_box)
            phase_positions[node] = best_position
            phase_rotations[node] = best_rotation
            placed_set.add(node)
        
        # Update current_pos for next phase (use this phase's results as starting point)
        current_pos = phase_positions.copy()
        final_positions = phase_positions.copy()
        node_rotations = phase_rotations.copy()
        
        # Calculate and print wire cost for this phase
        phase_wire_cost = calculate_total_wire_cost(phase_positions)
        print(f"  Phase {phase_idx + 1} complete. Total wire cost: {phase_wire_cost:.1f}")
    
    # Final pass: update graph with rotation info from the last phase
    for node in G.nodes():
        # Restore original dims/pins first
        G.nodes[node]['dims'] = original_node_props[node]['dims']
        G.nodes[node]['pin_locations'] = original_node_props[node]['pin_locations'].copy()
        
        # Apply final rotation
        best_rotation = node_rotations.get(node, 0)
        original_dims = G.nodes[node].get('dims', (1, 1, 1))
        ow, oh, od = int(original_dims[0]), int(original_dims[1]), int(original_dims[2])
        rotated_dims = rotate_dimensions(ow, oh, od, best_rotation)
        G.nodes[node]['dims'] = rotated_dims
        G.nodes[node]['rotation'] = best_rotation
        
        # Rotate pin locations to match the rotated node
        original_pins = original_node_props[node]['pin_locations']
        rotated_pins = rotate_pin_locations(original_pins, ow, oh, od, best_rotation)
        G.nodes[node]['pin_locations'] = rotated_pins
        
    final_wire_cost = calculate_total_wire_cost(final_positions)
    improvement = ((initial_wire_cost - final_wire_cost) / initial_wire_cost * 100) if initial_wire_cost > 0 else 0
    print(f"\nPlacement complete ({len(padding_phases)} phases).")
    print(f"Final wire cost: {final_wire_cost:.1f} (started at {initial_wire_cost:.1f}, {improvement:+.1f}% change)")
    return final_positions


def optimize_placement(G: nx.DiGraph, max_time_seconds: float = 60.0) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimizes component placement using a force-directed algorithm (Spring Layout)
    followed by a legalization step to resolve collisions and align to grid.
    """
    raw_pos = calculate_spring_layout(G, max_time_seconds)
    return legalize_placement(G, raw_pos)

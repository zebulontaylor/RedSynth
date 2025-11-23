from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import heapq
import random
import math

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
        for key in self._get_bucket_keys(x, y, z, w, h, d):
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(box)

    def query(self, x, y, z, w, h, d) -> bool:
        """Returns True if the box overlaps with any existing box in the index."""
        # Optimization: Avoid allocating set for checked_boxes
        # It is faster to re-check a few boxes than to allocate a set for every query
        
        min_bx = int(x // self.bucket_size)
        max_bx = int((x + w) // self.bucket_size)
        min_by = int(y // self.bucket_size)
        max_by = int((y + h) // self.bucket_size)
        min_bz = int(z // self.bucket_size)
        max_bz = int((z + d) // self.bucket_size)

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for bz in range(min_bz, max_bz + 1):
                    key = (bx, by, bz)
                    if key in self.buckets:
                        for obox in self.buckets[key]:
                            ox, oy, oz, ow, oh, od = obox
                            # Fast AABB check
                            if (x < ox + ow and x + w > ox and
                                y < oy + oh and y + h > oy and
                                z < oz + od and z + d > oz):
                                return True
        return False


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
    
    print("Legalizing placement (resolving collisions)...")
    
    node_info = {}
    for node in G.nodes():
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = int(dims[0]), int(dims[1]), int(dims[2])
        if 'pin_locations' not in G.nodes[node]:
            padding = 8
        else:
            padding = int(len(G.nodes[node]['pin_locations']) * 8 / h)
        node_info[node] = {
            'w': w, 'h': h, 'd': d,
            'w_padded': w + padding, 
            'h_padded': h,
            'd_padded': d + padding
        }
        
    final_positions: Dict[str, Tuple[float, float, float]] = {}
    node_rotations: Dict[str, int] = {}  # Store rotation for each node (0, 90, 180, 270)
    
    # Use SpatialIndex for fast collision detection
    spatial_index = SpatialIndex(bucket_size=20)
    
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
                cost += (abs(n_x - px) + abs(n_y - py) + abs(n_z - pz)) * 0.2
            else:
                continue
                
        return cost
        
    # Pre-calculate spiral offsets once
    print("Generating spiral offsets cache...")
    spiral_offsets_cache = []
    pq = [(0.0, 0.0, 0, 0, 0)]
    visited_offsets = {(0, 0, 0)}
    Y_PENALTY = 20.0
    MAX_OFFSETS = 1000000 # Limit search space
    
    while pq and len(spiral_offsets_cache) < MAX_OFFSETS:
        cost, _, dx, dy, dz = heapq.heappop(pq)
        spiral_offsets_cache.append((dx, dy, dz))
        
        for next_x, next_y, next_z in [
            (dx+1, dy, dz), (dx-1, dy, dz),
            (dx, dy+1, dz), (dx, dy-1, dz),
            (dx, dy, dz+1), (dx, dy, dz-1)
        ]:
            if (next_x, next_y, next_z) not in visited_offsets:
                visited_offsets.add((next_x, next_y, next_z))
                effective_y_penalty = Y_PENALTY * abs(next_y) / (abs(next_x) + abs(next_z) + 1)
                effective_y_penalty = max(effective_y_penalty - 0.25, 0.1)
                new_cost = next_x*next_x + effective_y_penalty * next_y*next_y + next_z*next_z
                heapq.heappush(pq, (new_cost, random.random(), next_x, next_y, next_z))

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
                wire_cost = calculate_wire_cost(node, (cx, cy, cz), all_neighbors, final_positions, raw_pos)
                
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
            
            # If cache was exhausted, we might need to search further or just stack it far away
            # For now, just use the last valid spot found or force place (shouldn't happen often with large cache)
            print(f"Warning: Could not find non-overlapping position for {node} within search limit.")
            fx, fy, fz = start_x, start_y, start_z
            cx = fx + w / 2.0
            cy = fy + h / 2.0
            cz = fz + d / 2.0
            best_position = (cx, cy, cz)
            best_box = (fx, fy, fz, w, h, d)
            best_rotation = 0
        
        spatial_index.insert(best_box)
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

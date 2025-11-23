from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import random
import math
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from .routing import RoutingGrid

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


def visualize_graph_interactive(G: nx.DiGraph, positions: Dict[str, Tuple[float, float, float]], routed_paths: Optional[Dict[str, List[List[Tuple[int, int, int]]]]] = None, failed_nets: Optional[List[Dict]] = None, grid: Optional[RoutingGrid] = None):
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
        
        x_min, x_max = int(round(px - pw/2)), int(round(px + pw/2))
        y_min, y_max = int(round(py - ph/2)), int(round(py + ph/2))
        z_min, z_max = int(round(pz - pd/2)), int(round(pz + pd/2))
        
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
            abs_x = int(round(x - w/2 + pin_ox))
            abs_y = int(round(y - h/2 + pin_oy))
            abs_z = int(round(z - d/2 + pin_oz))
            
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
    
    # Visualize Constants
    const_x = []
    const_y = []
    const_z = []
    const_text = []
    const_colors = []
    
    for node, (x, y, z) in positions.items():
        constants = G.nodes[node].get('constants', {})
        dims = G.nodes[node].get('dims', (1, 1, 1))
        w, h, d = dims
        pin_locations = G.nodes[node].get('pin_locations', {})
        
        for pin_name, val in constants.items():
             if pin_name in pin_locations:
                 px, py, pz = pin_locations[pin_name]
                 abs_x = int(round(x - w/2 + px))
                 abs_y = int(round(y - h/2 + py))
                 abs_z = int(round(z - d/2 + pz))
                 
                 # Calculate vector from center to pin
                 vx = abs_x - x
                 vy = abs_y - y
                 vz = abs_z - z
                 
                 # Normalize and apply offset
                 mag = math.sqrt(vx*vx + vy*vy + vz*vz)
                 offset = 0.8
                 if mag > 0:
                     off_x = (vx / mag) * offset
                     off_y = (vy / mag) * offset
                     off_z = (vz / mag) * offset
                 else:
                     off_x, off_y, off_z = 0, 0, 0
                 
                 const_x.append(abs_x + off_x)
                 const_y.append(abs_z + off_z) # Swap Y/Z
                 const_z.append(abs_y + off_y)
                 const_text.append(val)
                 const_colors.append('blue' if val == '0' else 'red')

    if const_x:
        fig.add_trace(go.Scatter3d(
            x=const_x,
            y=const_y,
            z=const_z,
            mode='text',
            text=const_text,
            textfont=dict(
                size=12,
                color=const_colors
            ),
            hoverinfo='none',
            name='Constants'
        ))
    
    # Highlight Failed Connections
    if failed_nets:
        fail_x, fail_y, fail_z = [], [], []
        fail_text = []
        for fail in failed_nets:
            start = fail['start']
            end = fail['end']
            net = fail['net']
            
            fail_x.extend([start[0], end[0]])
            fail_y.extend([start[2], end[2]]) # Swap Y/Z
            fail_z.extend([start[1], end[1]])
            fail_text.extend([f"Failed Start: {net}", f"Failed End: {net}"])
            
        if fail_x:
            fig.add_trace(go.Scatter3d(
                x=fail_x,
                y=fail_y,
                z=fail_z,
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                text=fail_text,
                hoverinfo='text',
                name='Failed Connections'
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

#!/usr/bin/env python3
import sys
import json
from redsynth import parse_netlist, build_graph, optimize_placement, route_nets, visualize_graph, visualize_2d_projection, visualize_graph_interactive
from redsynth.placement import calculate_spring_layout, legalize_placement


def save_routed_model(routed_paths, positions, G, output_file="routed_model.json"):
    """
    Save a successfully routed model to a JSON file.
    
    Args:
        routed_paths: Dict mapping net names to lists of paths (each path is a list of (x,y,z) tuples)
        positions: Dict mapping node names to (x,y,z) positions
        G: The networkx graph with node metadata
        output_file: Output filename
    """
    # Convert routed_paths (tuples are JSON-safe as lists)
    serialized_paths = {}
    for net_name, paths in routed_paths.items():
        serialized_paths[net_name] = [
            [list(point) for point in path]
            for path in paths
        ]
    
    # Convert positions
    serialized_positions = {}
    for node_name, pos in positions.items():
        serialized_positions[node_name] = list(pos)
    
    # Save node metadata from the graph
    nodes_metadata = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        nodes_metadata[node] = {
            'dims': list(node_data.get('dims', (1, 1, 1))),
            'cell_type': node_data.get('cell_type', 'unknown'),
            'pin_locations': {
                pin: list(loc) for pin, loc in node_data.get('pin_locations', {}).items()
            },
            'constants': node_data.get('constants', {})
        }
    
    model_data = {
        "routed_paths": serialized_paths,
        "positions": serialized_positions,
        "nodes_metadata": nodes_metadata
    }
    
    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"Saved routed model to {output_file}")


def load_routed_model(input_file="routed_model.json"):
    """
    Load a previously saved routed model from a JSON file.
    
    Args:
        input_file: Input filename
        
    Returns:
        Tuple of (routed_paths, positions, nodes_metadata)
        - routed_paths: Dict mapping net names to lists of paths
        - positions: Dict mapping node names to (x,y,z) tuples
        - nodes_metadata: Dict mapping node names to their metadata (dims, cell_type, pin_locations, constants)
    """
    with open(input_file, 'r') as f:
        model_data = json.load(f)
    
    # Convert routed_paths back to tuples
    routed_paths = {}
    for net_name, paths in model_data["routed_paths"].items():
        routed_paths[net_name] = [
            [tuple(point) for point in path]
            for path in paths
        ]
    
    # Convert positions back to tuples
    positions = {}
    for node_name, pos in model_data["positions"].items():
        positions[node_name] = tuple(pos)
    
    # Load nodes metadata and convert pin_locations back to tuples
    nodes_metadata = {}
    if "nodes_metadata" in model_data:
        for node_name, meta in model_data["nodes_metadata"].items():
            nodes_metadata[node_name] = {
                'dims': tuple(meta.get('dims', [1, 1, 1])),
                'cell_type': meta.get('cell_type', 'unknown'),
                'pin_locations': {
                    pin: tuple(loc) for pin, loc in meta.get('pin_locations', {}).items()
                },
                'constants': meta.get('constants', {})
            }
    
    print(f"Loaded routed model from {input_file}")
    print(f"  {len(routed_paths)} nets, {len(positions)} nodes")
    
    return routed_paths, positions, nodes_metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RedSynth - Redstone Circuit Synthesizer")
    parser.add_argument("netlist", nargs="?", default="netlist.json", help="Input netlist JSON file (default: netlist.json)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Optimization timeout in seconds (default: 60)")
    parser.add_argument("--load", metavar="MODEL_FILE", help="Load a previously saved routed model instead of synthesizing")
    parser.add_argument("--output", "-o", default="routed_model.json", help="Output file for saved model (default: routed_model.json)")
    parser.add_argument("--retry", action="store_true", help="Automatically restart placement and routing if connections fail")
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum number of retries when --retry is enabled (default: 10)")
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing routed model
        routed_paths, optimized_pos, nodes_metadata = load_routed_model(args.load)
        failed_nets = None
        
        # Regenerate block_grid from routed_paths (allows iterating on redstone conversion)
        print("Regenerating redstone block grid...")
        from redsynth.redstone import generate_redstone_grid
        block_grid, wire_positions = generate_redstone_grid(routed_paths)
        print(f"Generated {len(block_grid)} blocks.")
        
        # Create a minimal graph with node metadata for visualization
        import networkx as nx
        G = nx.DiGraph()
        for node_name, meta in nodes_metadata.items():
            G.add_node(node_name, **meta)
        
        print("Visualizing loaded model...")
        visualize_graph_interactive(G, optimized_pos, routed_paths=routed_paths, failed_nets=failed_nets, block_grid=block_grid, wire_positions=wire_positions)
    else:
        # Full synthesis flow
        netlist_file = args.netlist
        timeout = args.timeout
        
        print(f"Parsing netlist from {netlist_file}...")
        models = parse_netlist(netlist_file)
        
        print(f"\nFound {len(models)} nodes.\n")
            
        print("\nBuilding graph...")
        G = build_graph(models)
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        attempt = 0
        max_attempts = args.max_retries + 1 if args.retry else 1
        priority_nets = None  # Nets to prioritize in subsequent attempts
        
        while attempt < max_attempts:
            attempt += 1
            if args.retry:
                print(f"\n{'='*60}")
                print(f"Attempt {attempt}/{max_attempts}")
                print(f"{'='*60}\n")
            
            # Try optimization
            # optimized_pos = optimize_placement(G, max_time_seconds=timeout)
            
            print("Calculating spring layout...")
            # Use attempt number as seed to get different but reproducible layouts per attempt
            raw_pos = calculate_spring_layout(G, max_time_seconds=timeout, seed=attempt)
            
            print("Visualizing spring layout...")
            visualize_graph_interactive(G, raw_pos, output_filename="spring_layout_visualization.html")
            
            print("Legalizing placement...")
            optimized_pos = legalize_placement(G, raw_pos)
            
            routed_paths = None
            failed_nets = None
            grid = None
            
            if not optimized_pos:
                raise Exception("Optimization failed or timed out.")

            print("Using optimized placement.")
            # Run routing with priority nets from previous attempt
            # Enable early exit on failure if we have retries remaining (no point continuing after first failure)
            early_exit = args.retry and attempt < max_attempts
            routed_paths, failed_nets, grid, rip_up_counts = route_nets(G, optimized_pos, priority_nets=priority_nets, early_exit_on_failure=early_exit)
            
            # Check if routing succeeded
            if not failed_nets:
                if args.retry and attempt > 1:
                    print(f"\nRouting succeeded on attempt {attempt}!")
                break
            
            # Routing failed - collect priority nets for next attempt
            if args.retry and attempt < max_attempts:
                # Collect failed nets
                failed_net_names = set(f['net'] for f in failed_nets)
                # Collect nets that required rip-ups
                ripped_up_nets = set(net for net, count in rip_up_counts.items() if count > 0)
                # Combine for next attempt's priority
                priority_nets = [] #list(failed_net_names | ripped_up_nets)
                
                print(f"\n{len(failed_nets)} connection(s) failed to route.")
                if ripped_up_nets:
                    print(f"{len(ripped_up_nets)} net(s) required rip-ups.")
                print(f"Prioritizing {len(priority_nets)} net(s) for next attempt. Retrying...")
            elif args.retry:
                print(f"\n{len(failed_nets)} connection(s) failed after {max_attempts} attempts.")
        
        # Generate Redstone Grid
        print("Generating redstone block grid...")
        from redsynth.redstone import generate_redstone_grid
        block_grid, wire_positions = generate_redstone_grid(routed_paths)
        print(f"Generated {len(block_grid)} blocks.")
        
        visualize_graph(G, positions=optimized_pos, routed_paths=routed_paths)
        visualize_2d_projection(G, optimized_pos, routed_paths=routed_paths)
        visualize_graph_interactive(G, optimized_pos, routed_paths=routed_paths, failed_nets=failed_nets, grid=grid, block_grid=block_grid, wire_positions=wire_positions)
        
        # Save the routed model if all connections succeeded
        if not failed_nets:
            save_routed_model(routed_paths, optimized_pos, G, output_file=args.output)
        else:
            print(f"Model not saved: {len(failed_nets)} connection(s) failed to route.")

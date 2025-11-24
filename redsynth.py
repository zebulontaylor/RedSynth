#!/usr/bin/env python3
import sys
from redsynth import parse_netlist, build_graph, optimize_placement, route_nets, visualize_graph, visualize_2d_projection, visualize_graph_interactive
from redsynth.placement import calculate_spring_layout, legalize_placement

if __name__ == "__main__":
    # Default to netlist.json if no argument provided
    netlist_file = sys.argv[1] if len(sys.argv) > 1 else "netlist.json"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    
    print(f"Parsing netlist from {netlist_file}...")
    models = parse_netlist(netlist_file)
    
    print(f"\nFound {len(models)} nodes.\n")
        
    print("\nBuilding graph...")
    G = build_graph(models)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Try optimization
    # optimized_pos = optimize_placement(G, max_time_seconds=timeout)
    
    print("Calculating spring layout...")
    raw_pos = calculate_spring_layout(G, max_time_seconds=timeout)
    
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
    # Run routing
    routed_paths, failed_nets = route_nets(G, optimized_pos)
    
    # Generate Redstone Grid
    print("Generating redstone block grid...")
    from redsynth.redstone import generate_redstone_grid
    block_grid = generate_redstone_grid(routed_paths)
    print(f"Generated {len(block_grid)} blocks.")
    
    visualize_graph(G, positions=optimized_pos, routed_paths=routed_paths)
    visualize_2d_projection(G, optimized_pos, routed_paths=routed_paths)
    visualize_graph_interactive(G, optimized_pos, routed_paths=routed_paths, failed_nets=failed_nets, block_grid=block_grid)

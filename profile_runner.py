
import cProfile
import pstats
import redsynth as netlist_parser
import sys
import time

def profile_func(func, name, *args, **kwargs):
    print(f"Profiling {name}...")
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    profiler.disable()
    print(f"{name} took {end - start:.2f} seconds")
    
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
    return result

def main():
    netlist_file = "netlist.json"
    print(f"Parsing {netlist_file}...")
    models = netlist_parser.parse_netlist(netlist_file)
    
    print("Building graph...")
    G = netlist_parser.build_graph(models)
    
    # Profile Placement
    optimized_pos = profile_func(netlist_parser.optimize_placement, "optimize_placement", G, max_time_seconds=60.0)
    
    if optimized_pos:
        # Profile Routing
        # We'll wrap route_nets but maybe we want to catch it if it takes too long?
        # For now let's just run it.
        profile_func(netlist_parser.route_nets, "route_nets", G, optimized_pos, max_time_seconds=60.0)

if __name__ == "__main__":
    main()

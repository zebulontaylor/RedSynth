#!/usr/bin/env python3
"""
Visualization script for all possible redstone move / turn / intersection permutations.

This creates an interactive 3D HTML visualization showing:
- All cardinal moves (±X, ±Z flat)
- All slope moves (8 directions with vertical component)
- All turn combinations (cardinal→cardinal, cardinal→slope, slope→slope)
- Intersection patterns (two wires crossing)

Run: python tests/visualize_permutations.py
Output: permutation_visualization.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redsynth.redstone import generate_redstone_grid
from typing import Dict, List, Tuple

try:
    import plotly.graph_objects as go
except ImportError:
    print("Plotly is required. Install with: pip install plotly")
    sys.exit(1)


# ============================================================================
# Move Definitions (matching routing.py)
# ============================================================================

# Cardinals: 1 grid unit = 2 world units (flat horizontal moves)
CARDINALS = {
    "+X": (2, 0, 0),
    "-X": (-2, 0, 0),
    "+Z": (0, 0, 2),
    "-Z": (0, 0, -2),
}

# Slopes: 2 horizontal + 1 vertical (staircase moves)
SLOPES = {
    "+X+Y": (4, 2, 0),
    "+X-Y": (4, -2, 0),
    "-X+Y": (-4, 2, 0),
    "-X-Y": (-4, -2, 0),
    "+Z+Y": (0, 2, 4),
    "+Z-Y": (0, -2, 4),
    "-Z+Y": (0, 2, -4),
    "-Z-Y": (0, -2, -4),
}

ALL_MOVES = {**CARDINALS, **SLOPES}


def generate_path(start: Tuple[int, int, int], moves: List[str]) -> List[Tuple[int, int, int]]:
    """Generate a path from a starting point using a sequence of move names."""
    path = [start]
    current = start
    for move_name in moves:
        delta = ALL_MOVES[move_name]
        current = (current[0] + delta[0], current[1] + delta[1], current[2] + delta[2])
        path.append(current)
    return path


def create_test_cases() -> Dict[str, List[List[Tuple[int, int, int]]]]:
    """
    Create all test cases organized by category.
    Returns a dict mapping net names to lists of paths.
    """
    routed_paths = {}
    
    # Layout spacing
    section_spacing = 30
    row_spacing = 20
    
    # ========================================
    # Section 1: Single Moves (Cardinals)
    # ========================================
    base_y = 10
    base_x = 0
    base_z = 0
    
    for i, (name, delta) in enumerate(CARDINALS.items()):
        net_name = f"cardinal_{name}"
        start = (base_x + i * row_spacing, base_y, base_z)
        path = generate_path(start, [name, name, name])  # 3 moves in same direction
        routed_paths[net_name] = [path]
    
    # ========================================
    # Section 2: Single Moves (Slopes)
    # ========================================
    base_z = section_spacing
    
    for i, (name, delta) in enumerate(SLOPES.items()):
        net_name = f"slope_{name}"
        col = i % 4
        row = i // 4
        start = (base_x + col * row_spacing, base_y, base_z + row * row_spacing)
        path = generate_path(start, [name, name])  # 2 slope moves
        routed_paths[net_name] = [path]
    
    # ========================================
    # Section 3: Turns - Cardinal to Cardinal
    # ========================================
    base_z = section_spacing * 2
    
    cardinal_names = list(CARDINALS.keys())
    turn_idx = 0
    for i, c1 in enumerate(cardinal_names):
        for j, c2 in enumerate(cardinal_names):
            if c1 != c2 and c1 != c2.replace('+', '~').replace('-', '+').replace('~', '-'):  # Skip same axis
                if c1[1] != c2[1]:  # Different axis (X vs Z)
                    net_name = f"turn_card_{c1}_{c2}"
                    col = turn_idx % 4
                    row = turn_idx // 4
                    start = (base_x + col * row_spacing, base_y, base_z + row * row_spacing)
                    path = generate_path(start, [c1, c1, c2, c2])
                    routed_paths[net_name] = [path]
                    turn_idx += 1
    
    # ========================================
    # Section 4: Turns - Cardinal to Slope
    # ========================================
    base_z = section_spacing * 3
    
    # Sample of cardinal → slope turns (not exhaustive to keep visualization manageable)
    card_slope_turns = [
        (["+X", "+X"], ["+Z+Y"]),
        (["+X", "+X"], ["+Z-Y"]),
        (["-X", "-X"], ["-Z+Y"]),
        (["+Z", "+Z"], ["+X+Y"]),
        (["+Z", "+Z"], ["-X-Y"]),
        (["-Z", "-Z"], ["+X-Y"]),
    ]
    
    for i, (card_moves, slope_moves) in enumerate(card_slope_turns):
        net_name = f"turn_card_slope_{i}"
        col = i % 3
        row = i // 3
        start = (base_x + col * 24, base_y, base_z + row * 24)
        path = generate_path(start, card_moves + slope_moves)
        routed_paths[net_name] = [path]
    
    # ========================================
    # Section 5: Turns - Slope to Cardinal
    # ========================================
    base_z = section_spacing * 4
    
    slope_card_turns = [
        (["+X+Y"], ["+Z", "+Z"]),
        (["+X-Y"], ["-Z", "-Z"]),
        (["-X+Y"], ["+Z", "+Z"]),
        (["+Z+Y"], ["+X", "+X"]),
        (["+Z-Y"], ["-X", "-X"]),
        (["-Z+Y"], ["+X", "+X"]),
    ]
    
    for i, (slope_moves, card_moves) in enumerate(slope_card_turns):
        net_name = f"turn_slope_card_{i}"
        col = i % 3
        row = i // 3
        start = (base_x + col * 24, base_y, base_z + row * 24)
        path = generate_path(start, slope_moves + card_moves)
        routed_paths[net_name] = [path]
    
    # ========================================
    # Section 6: Turns - Slope to Slope
    # ========================================
    base_z = section_spacing * 5
    
    slope_slope_turns = [
        (["+X+Y"], ["+Z+Y"]),
        (["+X+Y"], ["+Z-Y"]),
        (["+X-Y"], ["+Z+Y"]),
        (["-X+Y"], ["-Z+Y"]),
        (["+Z+Y"], ["+X+Y"]),
        (["+Z-Y"], ["-X-Y"]),
    ]
    
    for i, (s1, s2) in enumerate(slope_slope_turns):
        net_name = f"turn_slope_slope_{i}"
        col = i % 3
        row = i // 3
        start = (base_x + col * 24, base_y, base_z + row * 24)
        path = generate_path(start, s1 + s2)
        routed_paths[net_name] = [path]
    
    # ========================================
    # Section 7: Intersections (Crossings)
    # Wires crossing at different Y levels (one above the other)
    # ========================================
    base_z = section_spacing * 6
    
    # Intersection 1: X-wire below, Z-wire above (crossing at center)
    # Z-wire at Y=base_y+1, X-wire at Y=base_y
    int_x_start = (base_x, base_y, base_z + 6)
    int_x_path = generate_path(int_x_start, ["+X", "+X", "+X", "+X", "+X"])
    routed_paths["intersection_1_x_wire"] = [int_x_path]
    
    int_z_start = (base_x + 4, base_y, base_z)
    int_z_path = generate_path(int_z_start, ["+Z", "+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_1_z_wire"] = [int_z_path]
    
    # Intersection 2: Z-wire below, X-wire above
    offset2 = 30
    int2_z_start = (base_x + offset2 + 6, base_y, base_z)
    int2_z_path = generate_path(int2_z_start, ["+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_2_z_wire"] = [int2_z_path]
    
    int2_x_start = (base_x + offset2, base_y, base_z + 4)
    int2_x_path = generate_path(int2_x_start, ["+X", "+X", "+X", "+X", "+X", "+X"])
    routed_paths["intersection_2_x_wire"] = [int2_x_path]
    
    # Intersection 3: Multiple crossings on same X-wire
    offset3 = 60
    int3_x_start = (base_x + offset3, base_y, base_z + 6)
    int3_x_path = generate_path(int3_x_start, ["+X", "+X", "+X", "+X", "+X", "+X", "+X", "+X"])
    routed_paths["intersection_3_x_wire"] = [int3_x_path]
    
    # Two Z-wires crossing the X-wire
    int3_z1_start = (base_x + offset3 + 4, base_y, base_z)
    int3_z1_path = generate_path(int3_z1_start, ["+Z", "+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_3_z_wire_1"] = [int3_z1_path]
    
    int3_z2_start = (base_x + offset3 + 12, base_y, base_z)
    int3_z2_path = generate_path(int3_z2_start, ["+Z", "+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_3_z_wire_2"] = [int3_z2_path]
    
    # Intersection 4: Grid pattern (2x2 intersections)
    offset4 = 100
    # Two X-wires
    int4_x1_start = (base_x + offset4, base_y, base_z + 4)
    int4_x1_path = generate_path(int4_x1_start, ["+X", "+X", "+X", "+X", "+X", "+X"])
    routed_paths["intersection_4_x_wire_1"] = [int4_x1_path]
    
    int4_x2_start = (base_x + offset4, base_y, base_z + 10)
    int4_x2_path = generate_path(int4_x2_start, ["+X", "+X", "+X", "+X", "+X", "+X"])
    routed_paths["intersection_4_x_wire_2"] = [int4_x2_path]
    
    # Two Z-wires crossing both X-wires
    int4_z1_start = (base_x + offset4 + 4, base_y, base_z)
    int4_z1_path = generate_path(int4_z1_start, ["+Z", "+Z", "+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_4_z_wire_1"] = [int4_z1_path]
    
    int4_z2_start = (base_x + offset4 + 10, base_y, base_z)
    int4_z2_path = generate_path(int4_z2_start, ["+Z", "+Z", "+Z", "+Z", "+Z", "+Z", "+Z"])
    routed_paths["intersection_4_z_wire_2"] = [int4_z2_path]
    
    # ========================================
    # Section 8: Complex Patterns
    # ========================================
    base_z = section_spacing * 7
    
    # Zigzag pattern
    start7 = (base_x, base_y, base_z)
    path7 = generate_path(start7, ["+X", "+X", "+Z", "+Z", "+X", "+X", "+Z", "+Z"])
    routed_paths["zigzag_pattern"] = [path7]
    
    # Staircase up then down
    start8 = (base_x + 50, base_y, base_z)
    path8 = generate_path(start8, ["+X+Y", "+X+Y", "+X+Y", "+X-Y", "+X-Y", "+X-Y"])
    routed_paths["staircase_up_down"] = [path8]
    
    # Spiral-ish pattern
    start9 = (base_x + 100, base_y, base_z)
    path9 = generate_path(start9, ["+X+Y", "+Z", "+Z", "-X+Y", "-Z", "-Z", "+X+Y"])
    routed_paths["spiral_pattern"] = [path9]
    
    # ========================================
    # Section 9: STACKED Cardinals (2 wires stacked vertically)
    # ========================================
    base_z = section_spacing * 8
    stack_offset = 2  # Y offset between stacked wires
    
    for i, (name, delta) in enumerate(CARDINALS.items()):
        # Bottom wire
        net_name_1 = f"stacked_cardinal_{name}_bottom"
        start1 = (base_x + i * row_spacing, base_y, base_z)
        path1 = generate_path(start1, [name, name, name])
        routed_paths[net_name_1] = [path1]
        
        # Top wire (stacked above)
        net_name_2 = f"stacked_cardinal_{name}_top"
        start2 = (base_x + i * row_spacing, base_y + stack_offset, base_z)
        path2 = generate_path(start2, [name, name, name])
        routed_paths[net_name_2] = [path2]
    
    # ========================================
    # Section 10: STACKED Slopes (2 wires stacked vertically)
    # ========================================
    base_z = section_spacing * 9
    
    for i, (name, delta) in enumerate(SLOPES.items()):
        col = i % 4
        row = i // 4
        
        # Bottom wire
        net_name_1 = f"stacked_slope_{name}_bottom"
        start1 = (base_x + col * row_spacing, base_y, base_z + row * row_spacing)
        path1 = generate_path(start1, [name, name])
        routed_paths[net_name_1] = [path1]
        
        # Top wire (stacked above)
        net_name_2 = f"stacked_slope_{name}_top"
        start2 = (base_x + col * row_spacing, base_y + stack_offset, base_z + row * row_spacing)
        path2 = generate_path(start2, [name, name])
        routed_paths[net_name_2] = [path2]
    
    # ========================================
    # Section 11: STACKED Cardinal→Cardinal Turns
    # ========================================
    base_z = section_spacing * 10
    
    turn_idx = 0
    for i, c1 in enumerate(cardinal_names):
        for j, c2 in enumerate(cardinal_names):
            if c1 != c2 and c1 != c2.replace('+', '~').replace('-', '+').replace('~', '-'):
                if c1[1] != c2[1]:
                    col = turn_idx % 4
                    row = turn_idx // 4
                    
                    # Bottom wire
                    net_name_1 = f"stacked_turn_card_{c1}_{c2}_bottom"
                    start1 = (base_x + col * row_spacing, base_y, base_z + row * row_spacing)
                    path1 = generate_path(start1, [c1, c1, c2, c2])
                    routed_paths[net_name_1] = [path1]
                    
                    # Top wire
                    net_name_2 = f"stacked_turn_card_{c1}_{c2}_top"
                    start2 = (base_x + col * row_spacing, base_y + stack_offset, base_z + row * row_spacing)
                    path2 = generate_path(start2, [c1, c1, c2, c2])
                    routed_paths[net_name_2] = [path2]
                    
                    turn_idx += 1
    
    # ========================================
    # Section 12: STACKED Cardinal→Slope Turns
    # ========================================
    base_z = section_spacing * 11
    
    for i, (card_moves, slope_moves) in enumerate(card_slope_turns):
        col = i % 3
        row = i // 3
        
        # Bottom wire
        net_name_1 = f"stacked_turn_card_slope_{i}_bottom"
        start1 = (base_x + col * 24, base_y, base_z + row * 24)
        path1 = generate_path(start1, card_moves + slope_moves)
        routed_paths[net_name_1] = [path1]
        
        # Top wire
        net_name_2 = f"stacked_turn_card_slope_{i}_top"
        start2 = (base_x + col * 24, base_y + stack_offset, base_z + row * 24)
        path2 = generate_path(start2, card_moves + slope_moves)
        routed_paths[net_name_2] = [path2]
    
    # ========================================
    # Section 13: STACKED Slope→Cardinal Turns
    # ========================================
    base_z = section_spacing * 12
    
    for i, (slope_moves, card_moves) in enumerate(slope_card_turns):
        col = i % 3
        row = i // 3
        
        # Bottom wire
        net_name_1 = f"stacked_turn_slope_card_{i}_bottom"
        start1 = (base_x + col * 24, base_y, base_z + row * 24)
        path1 = generate_path(start1, slope_moves + card_moves)
        routed_paths[net_name_1] = [path1]
        
        # Top wire
        net_name_2 = f"stacked_turn_slope_card_{i}_top"
        start2 = (base_x + col * 24, base_y + stack_offset, base_z + row * 24)
        path2 = generate_path(start2, slope_moves + card_moves)
        routed_paths[net_name_2] = [path2]
    
    # ========================================
    # Section 14: STACKED Slope→Slope Turns
    # ========================================
    base_z = section_spacing * 13
    
    for i, (s1, s2) in enumerate(slope_slope_turns):
        col = i % 3
        row = i // 3
        
        # Bottom wire
        net_name_1 = f"stacked_turn_slope_slope_{i}_bottom"
        start1 = (base_x + col * 24, base_y, base_z + row * 24)
        path1 = generate_path(start1, s1 + s2)
        routed_paths[net_name_1] = [path1]
        
        # Top wire
        net_name_2 = f"stacked_turn_slope_slope_{i}_top"
        start2 = (base_x + col * 24, base_y + stack_offset, base_z + row * 24)
        path2 = generate_path(start2, s1 + s2)
        routed_paths[net_name_2] = [path2]
    
    # ========================================
    # Section 15: STACKED Complex Patterns
    # ========================================
    base_z = section_spacing * 14
    
    # Stacked Zigzag
    start_zz1 = (base_x, base_y, base_z)
    path_zz1 = generate_path(start_zz1, ["+X", "+X", "+Z", "+Z", "+X", "+X", "+Z", "+Z"])
    routed_paths["stacked_zigzag_bottom"] = [path_zz1]
    
    start_zz2 = (base_x, base_y + stack_offset, base_z)
    path_zz2 = generate_path(start_zz2, ["+X", "+X", "+Z", "+Z", "+X", "+X", "+Z", "+Z"])
    routed_paths["stacked_zigzag_top"] = [path_zz2]
    
    # Stacked Staircase
    start_st1 = (base_x + 50, base_y, base_z)
    path_st1 = generate_path(start_st1, ["+X+Y", "+X+Y", "+X+Y", "+X-Y", "+X-Y", "+X-Y"])
    routed_paths["stacked_staircase_bottom"] = [path_st1]
    
    start_st2 = (base_x + 50, base_y + stack_offset, base_z)
    path_st2 = generate_path(start_st2, ["+X+Y", "+X+Y", "+X+Y", "+X-Y", "+X-Y", "+X-Y"])
    routed_paths["stacked_staircase_top"] = [path_st2]
    
    # ========================================
    # Section 16: Signal Decay / Repeater Insertion Tests
    # Redstone signal decays 1 level per block (max 15).
    # Repeaters should be inserted before signal dies.
    # ========================================
    base_z = section_spacing * 15
    
    # ---- Boundary Tests: Around 15-block limit ----
    # Each cardinal move is 2 blocks, so:
    # 7 moves = 14 blocks (no repeater needed)
    # 8 moves = 16 blocks (1 repeater needed)
    # 15 moves = 30 blocks (2 repeaters needed)
    
    # 14 blocks - just under limit (NO repeater expected)
    start_14 = (base_x, base_y, base_z)
    path_14 = generate_path(start_14, ["+X"] * 7)
    routed_paths["decay_14_blocks_no_repeater"] = [path_14]
    
    # 16 blocks - just over limit (1 repeater expected)
    start_16 = (base_x, base_y, base_z + 10)
    path_16 = generate_path(start_16, ["+X"] * 8)
    routed_paths["decay_16_blocks_1_repeater"] = [path_16]
    
    # 30 blocks - double the limit (2 repeaters expected)
    start_30 = (base_x, base_y, base_z + 20)
    path_30 = generate_path(start_30, ["+X"] * 15)
    routed_paths["decay_30_blocks_2_repeaters"] = [path_30]
    
    # 46 blocks - triple limit (3 repeaters expected)
    start_46 = (base_x, base_y, base_z + 30)
    path_46 = generate_path(start_46, ["+X"] * 23)
    routed_paths["decay_46_blocks_3_repeaters"] = [path_46]
    
    # ---- Long Winding Wires ----
    # Turns shouldn't reset signal decay
    
    # Winding 16 blocks (1 repeater)
    start_wind1 = (base_x, base_y, base_z + 50)
    path_wind1 = generate_path(start_wind1, ["+X", "+X", "+Z", "+Z", "+X", "+X", "+Z", "+Z"])
    routed_paths["decay_winding_16_blocks"] = [path_wind1]
    
    # Winding 30 blocks (2 repeaters)
    start_wind2 = (base_x, base_y, base_z + 70)
    path_wind2 = generate_path(start_wind2, ["+X", "+X", "+Z", "+Z"] * 4 + ["-X"])
    routed_paths["decay_winding_30_blocks"] = [path_wind2]
    
    # Zigzag 40 blocks
    start_zig = (base_x, base_y, base_z + 100)
    path_zig = generate_path(start_zig, ["+X", "+Z", "+X", "+Z", "+X", "+Z", "+X", "+Z",
                                          "+X", "+Z", "+X", "+Z", "+X", "+Z", "+X", "+Z",
                                          "+X", "+Z", "+X", "+Z"])
    routed_paths["decay_zigzag_40_blocks"] = [path_zig]
    
    # ---- Long Slope Wires ----
    # Slopes are 4 horizontal blocks per move
    # 4 slope moves = 16 blocks (1 repeater)
    # 8 slope moves = 32 blocks (2+ repeaters)
    
    # 16 blocks up slope
    start_slope16 = (base_x + 120, base_y, base_z)
    path_slope16 = generate_path(start_slope16, ["+X+Y"] * 4)
    routed_paths["decay_slope_up_16_blocks"] = [path_slope16]
    
    # 16 blocks down slope
    start_slope16d = (base_x + 120, base_y + 10, base_z + 20)
    path_slope16d = generate_path(start_slope16d, ["+X-Y"] * 4)
    routed_paths["decay_slope_down_16_blocks"] = [path_slope16d]
    
    # 32 blocks slope staircase
    start_slope32 = (base_x + 120, base_y, base_z + 40)
    path_slope32 = generate_path(start_slope32, ["+X+Y"] * 8)
    routed_paths["decay_slope_32_blocks"] = [path_slope32]
    
    # Up then down staircase (32 blocks total)
    start_updown = (base_x + 120, base_y, base_z + 80)
    path_updown = generate_path(start_updown, ["+X+Y"] * 4 + ["+X-Y"] * 4)
    routed_paths["decay_slope_up_down_32_blocks"] = [path_updown]
    
    # ---- Mixed Cardinal + Slope ----
    # Tests decay across mixed move types
    
    # Cardinal then slope (24 blocks: 8 cardinal + 16 slope)
    start_mix1 = (base_x, base_y, base_z + 130)
    path_mix1 = generate_path(start_mix1, ["+X"] * 4 + ["+Z+Y"] * 4)
    routed_paths["decay_mixed_cardinal_slope_24"] = [path_mix1]
    
    # Slope then cardinal (24 blocks)
    start_mix2 = (base_x + 50, base_y, base_z + 130)
    path_mix2 = generate_path(start_mix2, ["+X+Y"] * 4 + ["+Z"] * 4)
    routed_paths["decay_mixed_slope_cardinal_24"] = [path_mix2]
    
    # Complex mixed (40 blocks)
    start_mix3 = (base_x + 100, base_y, base_z + 130)
    path_mix3 = generate_path(start_mix3, ["+X", "+X", "+X+Y", "+X+Y", "+Z", "+Z", 
                                            "+Z-Y", "+Z-Y", "+X", "+X", "+X+Y", "+X"])
    routed_paths["decay_mixed_complex_40"] = [path_mix3]
    
    # ---- Very Long Wires (Stress Tests) ----
    # These should require many repeaters
    
    # 60 blocks straight (4 repeaters)
    start_60 = (base_x, base_y, base_z + 160)
    path_60 = generate_path(start_60, ["+X"] * 30)
    routed_paths["decay_60_blocks_4_repeaters"] = [path_60]
    
    # 100 blocks straight (7 repeaters) - very long wire test
    start_100 = (base_x, base_y, base_z + 175)
    path_100 = generate_path(start_100, ["+X"] * 50)
    routed_paths["decay_100_blocks_7_repeaters"] = [path_100]
    
    # Spiral staircase (long path going around)
    start_spiral = (base_x, base_y, base_z + 200)
    path_spiral = generate_path(start_spiral, ["+X+Y", "+X+Y", "+Z", "+Z", "+Z+Y", "+Z+Y",
                                                "-X", "-X", "-X+Y", "-X+Y", "-Z", "-Z",
                                                "-Z+Y", "-Z+Y", "+X", "+X", "+X+Y", "+X+Y"])
    routed_paths["decay_spiral_staircase_long"] = [path_spiral]
    
    # ---- Edge Cases ----
    
    # Exact 15 blocks (at limit - borderline)
    # Note: Can't get exactly 15 with cardinal (2-block) moves
    # Using 7 cardinals = 14 + something extra
    start_exact = (base_x + 150, base_y, base_z + 160)
    path_exact = generate_path(start_exact, ["+X"] * 7 + ["+Z"])  # 14 + 2 = 16
    routed_paths["decay_exact_boundary_16"] = [path_exact]
    
    # Multi-direction long path (tests decay across many turns)
    start_multi = (base_x + 150, base_y, base_z + 180)
    path_multi = generate_path(start_multi, ["+X", "+Z", "-X", "+Z", "+X", "+Z", "-X", "+Z",
                                              "+X", "+Z", "-X", "+Z", "+X", "+Z", "-X", "+Z"])
    routed_paths["decay_serpentine_32_blocks"] = [path_multi]

    return routed_paths


def create_visualization(routed_paths: Dict[str, List[List[Tuple[int, int, int]]]], 
                         block_grid: Dict[Tuple[int, int, int], str],
                         output_file: str = "permutation_visualization.html"):
    """Create an interactive Plotly visualization."""
    
    print(f"Creating visualization with {len(routed_paths)} nets and {len(block_grid)} blocks...")
    
    fig = go.Figure()
    
    # Block styles
    BLOCK_STYLES = {
        "redstone_wire": {"color": "red", "opacity": 0.9, "name": "Redstone Dust", "ymin": 0.0, "ymax": 0.3},
        "stone": {"color": "gray", "opacity": 1.0, "name": "Stone"},
        "glass": {"color": "lightblue", "opacity": 0.4, "name": "Glass"},
        "repeater": {"color": "gold", "opacity": 0.95, "name": "Repeater", "ymin": 0.0, "ymax": 0.4},
        "default": {"color": "purple", "opacity": 0.5, "name": "Unknown Block"}
    }
    
    # Group blocks by type
    blocks_by_type = {}
    for coord, block_type in block_grid.items():
        if block_type not in blocks_by_type:
            blocks_by_type[block_type] = []
        blocks_by_type[block_type].append(coord)
    
    # Render each block type as a separate trace
    for block_type, coords in blocks_by_type.items():
        base_type = block_type.split(':')[0]
        style = BLOCK_STYLES.get(base_type, BLOCK_STYLES["default"])
        
        b_x, b_y, b_z = [], [], []
        b_i, b_j, b_k = [], [], []
        
        current_idx = 0
        
        for (x, y, z) in coords:
            # Swap Y and Z for visualization (Y is height in Minecraft, but Z is up in Plotly default)
            px, py, pz = x, z, y
            w, h, d = 1, 1, 1
            
            x_min, x_max = px + 0, px + w
            y_min, y_max = py + 0, py + d
            z_min, z_max = pz + style.get("ymin", 0), pz + style.get("ymax", h)
            
            # Vertices
            b_x.extend([x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min])
            b_y.extend([y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max])
            b_z.extend([z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max])
            
            # 12 triangles (6 faces, 2 triangles each)
            # Front
            b_i.extend([current_idx+0, current_idx+0])
            b_j.extend([current_idx+1, current_idx+2])
            b_k.extend([current_idx+2, current_idx+3])
            # Back
            b_i.extend([current_idx+4, current_idx+4])
            b_j.extend([current_idx+5, current_idx+6])
            b_k.extend([current_idx+6, current_idx+7])
            # Bottom
            b_i.extend([current_idx+0, current_idx+0])
            b_j.extend([current_idx+1, current_idx+5])
            b_k.extend([current_idx+5, current_idx+4])
            # Top
            b_i.extend([current_idx+2, current_idx+2])
            b_j.extend([current_idx+3, current_idx+7])
            b_k.extend([current_idx+7, current_idx+6])
            # Left
            b_i.extend([current_idx+0, current_idx+0])
            b_j.extend([current_idx+3, current_idx+7])
            b_k.extend([current_idx+7, current_idx+4])
            # Right
            b_i.extend([current_idx+1, current_idx+1])
            b_j.extend([current_idx+2, current_idx+6])
            b_k.extend([current_idx+6, current_idx+5])
            
            current_idx += 8
        
        fig.add_trace(go.Mesh3d(
            x=b_x, y=b_y, z=b_z,
            i=b_i, j=b_j, k=b_k,
            color=style["color"],
            opacity=style["opacity"],
            name=style["name"],
            flatshading=True,
            legendgroup=block_type,
            showlegend=True
        ))
    
    # Draw wire paths as lines
    edge_x, edge_y, edge_z = [], [], []
    
    for net_name, paths in routed_paths.items():
        for path in paths:
            for p in path:
                edge_x.append(p[0])
                edge_y.append(p[2])  # Swap Y/Z
                edge_z.append(p[1])
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)
    
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='darkblue', width=4),
        opacity=0.7,
        name='Wire Paths'
    ))
    
    # Add labels for sections (Z = section_spacing * section_number, spacing = 30)
    labels = [
        ((30, 10, 0), "Cardinals"),
        ((30, 10, 30), "Slopes"),
        ((30, 10, 60), "Cardinal→Cardinal Turns"),
        ((30, 10, 90), "Cardinal→Slope Turns"),
        ((30, 10, 120), "Slope→Cardinal Turns"),
        ((30, 10, 150), "Slope→Slope Turns"),
        ((70, 10, 180), "✕ Intersections"),
        ((50, 10, 210), "Complex Patterns"),
        # Stacked sections
        ((30, 12, 240), "⬆ STACKED Cardinals"),
        ((30, 12, 270), "⬆ STACKED Slopes"),
        ((30, 12, 300), "⬆ STACKED Card→Card Turns"),
        ((30, 12, 330), "⬆ STACKED Card→Slope Turns"),
        ((30, 12, 360), "⬆ STACKED Slope→Card Turns"),
        ((30, 12, 390), "⬆ STACKED Slope→Slope Turns"),
        ((50, 12, 420), "⬆ STACKED Complex Patterns"),
        # Signal decay / repeater tests (section 16 at Z=450)
        ((60, 10, 450), "🔁 SIGNAL DECAY - Boundary Tests"),
        ((60, 10, 500), "🔁 SIGNAL DECAY - Winding Wires"),
        ((150, 10, 450), "🔁 SIGNAL DECAY - Slopes"),
        ((60, 10, 580), "🔁 SIGNAL DECAY - Mixed Types"),
        ((60, 10, 610), "🔁 SIGNAL DECAY - Very Long Wires"),
        ((180, 10, 610), "🔁 SIGNAL DECAY - Edge Cases"),
    ]
    
    label_x = [l[0][0] for l in labels]
    label_y = [l[0][2] for l in labels]  # Swap Y/Z
    label_z = [l[0][1] + 5 for l in labels]
    label_text = [l[1] for l in labels]
    
    fig.add_trace(go.Scatter3d(
        x=label_x,
        y=label_y,
        z=label_z,
        mode='text',
        text=label_text,
        textfont=dict(size=14, color='black'),
        name='Section Labels',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Redstone Move/Turn/Intersection Permutations",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Z (Depth)',
            zaxis_title='Y (Height)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    print(f"Saving visualization to {output_file}...")
    fig.write_html(output_file)
    print("Done!")


def main():
    print("=" * 60)
    print("Redstone Permutation Visualization Generator")
    print("=" * 60)
    
    # Generate all test cases
    print("\nGenerating test paths...")
    routed_paths = create_test_cases()
    print(f"  Created {len(routed_paths)} test nets")
    
    # Count total path segments
    total_segments = sum(
        len(path) - 1 
        for paths in routed_paths.values() 
        for path in paths
    )
    print(f"  Total path segments: {total_segments}")
    
    # Generate redstone grid
    print("\nGenerating redstone blocks...")
    block_grid = generate_redstone_grid(routed_paths)
    print(f"  Generated {len(block_grid)} blocks")
    
    # Count by type
    type_counts = {}
    for block_type in block_grid.values():
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    for bt, count in sorted(type_counts.items()):
        print(f"    {bt}: {count}")
    
    # Create visualization
    print("\nCreating visualization...")
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "permutation_visualization.html")
    create_visualization(routed_paths, block_grid, output_path)
    
    print(f"\n✓ Open {output_path} in a browser to view the visualization!")


if __name__ == "__main__":
    main()


import json
from typing import List, Dict, Tuple
import networkx as nx
from .models import Node

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
    try:
        with open(cells_json_path, 'r') as f:
            cell_layouts = json.load(f)
    except FileNotFoundError:
        pass # It's okay if the file doesn't exist, we'll generate dynamic layouts
    
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
                
                num_bits = len(bits)
                height = max(2, num_bits * 2)
                
                node = Node(name=port_name, cell_type='InputPort' if direction == 'input' else 'OutputPort')
                node.set_bounding_box(0, 0, 0, 2, height, 2)
                
                if direction == 'input':
                    for i, bit in enumerate(bits):
                        # Pin locations: x=2 (right side), y=1, 3, 5... (centered in block), z=1
                        y_pos = 1 + i * 2
                        pin_name = f"out[{i}]" if num_bits > 1 else "out"
                        node.set_pin_location(pin_name, 2, y_pos, 1)
                        
                        if isinstance(bit, int) and bit >= 0:
                            node.add_outgoing_connection(pin_name, f"net_{bit}")
                else:
                    for i, bit in enumerate(bits):
                        # Pin locations: x=0 (left side), y=1, 3, 5... (centered in block), z=1
                        y_pos = 1 + i * 2
                        pin_name = f"in[{i}]" if num_bits > 1 else "in"
                        node.set_pin_location(pin_name, 0, y_pos, 1)
                        
                        if isinstance(bit, int) and bit >= 0:
                            node.add_incoming_connection(pin_name, f"net_{bit}")
                
                nodes.append(node)

        for cell_name, cell_data in cells.items():
            cell_type = cell_data.get('type', 'unknown')
            node = Node(name=cell_name, cell_type=cell_type)
            
            layout_data = cell_layouts.get(cell_type, {})
            
            # Generic dynamic layout for unknown cells
            if not layout_data:
                # print(f"Generating dynamic layout for unknown cell type: {cell_type}")
                connections = cell_data.get('connections', {})
                port_directions = cell_data.get('port_directions', {})
                
                input_ports = {} # port_name -> [pin_names]
                output_pins = []
                
                for port, bits in connections.items():
                    direction = port_directions.get(port, 'input')
                    if direction not in ['input', 'output']:
                         if port in ['Y', 'Q', 'OUT', 'DO']: direction = 'output'
                         else: direction = 'input'

                    if direction == 'input':
                        pin_names = []
                        for i in range(len(bits)):
                            pin_names.append(f"{port}[{i}]" if len(bits) > 1 else port)
                        input_ports[port] = pin_names
                    else:
                        for i in range(len(bits)):
                            pin_name = f"{port}[{i}]" if len(bits) > 1 else port
                            output_pins.append(pin_name)
                            
                # Calculate dimensions
                num_input_ports = len(input_ports)
                max_input_bits = 0
                for pins in input_ports.values():
                    max_input_bits = max(max_input_bits, len(pins))
                    
                max_pins_vertical = max(max_input_bits, len(output_pins))
                
                height = max(2, max_pins_vertical * 2)
                
                # Width: enough for inputs spread by 2 (x=0, x=2, ...), plus space for outputs
                last_input_x = (num_input_ports - 1) * 2 if num_input_ports > 0 else 0
                width = max(4, last_input_x + 3) 
                depth = 2
                
                layout_data = {
                    'bbox': {'width': width, 'height': height, 'depth': depth},
                    'inputs': {},
                    'outputs': {}
                }
                
                # Place inputs
                sorted_ports = sorted(input_ports.keys())
                for i, port in enumerate(sorted_ports):
                    x_pos = i * 2
                    pins = input_ports[port]
                    for bit_idx, pin_name in enumerate(pins):
                        # Align bits: bit 0 at y=1, bit 1 at y=3...
                        y_pos = 1 + bit_idx * 2
                        layout_data['inputs'][pin_name] = {'x': x_pos, 'y': y_pos, 'z': 0}
                        
                # Place outputs on right
                for i, pin in enumerate(output_pins):
                    y_pos = int((i + 0.5) * (height / len(output_pins))) if output_pins else 1
                    layout_data['outputs'][pin] = {'x': width, 'y': y_pos, 'z': 0}

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
                        elif isinstance(bit, (int, str)) and str(bit) in ['0', '1']:
                             pin_key = f"{port_name}[{i}]" if is_array else port_name
                             node.add_constant_connection(pin_key, str(bit))
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
        G.add_node(node.name, cell_type=node.cell_type, dims=dims, port_count=port_count, pin_locations=node.pin_locations, constants=node.constants)
        
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

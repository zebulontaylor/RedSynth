from typing import Dict, List, Tuple, Optional

class Node:
    """Represents a node in the netlist with bounding box and connections."""
    
    def __init__(self, name: str, cell_type: str = None):
        self.name = name
        self.cell_type = cell_type
        self.bounding_box: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
        self.incoming: Dict[str, List[str]] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.pin_locations: Dict[str, Tuple[int, int, int]] = {}
        self.constants: Dict[str, str] = {}
    
    def set_bounding_box(self, x: int, y: int, z: int, width: int, height: int, depth: int):
        self.bounding_box = (x, y, z, width, height, depth)
    
    def add_incoming_connection(self, port: str, connection_key: str):
        if port not in self.incoming:
            self.incoming[port] = []
        self.incoming[port].append(connection_key)
    
    def add_outgoing_connection(self, port: str, connection_key: str):
        if port not in self.outgoing:
            self.outgoing[port] = []
        self.outgoing[port].append(connection_key)
        
    def set_pin_location(self, port: str, x: int, y: int, z: int):
        self.pin_locations[port] = (x, y, z)

    def add_constant_connection(self, port: str, value: str):
        self.constants[port] = value
    
    def __repr__(self):
        return (f"Node(name={self.name!r}, type={self.cell_type!r}, "
                f"bbox={self.bounding_box}, "
                f"in={list(self.incoming.keys())}, "
                f"out={list(self.outgoing.keys())})")

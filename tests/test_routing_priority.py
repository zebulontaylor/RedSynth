
import unittest
from redsynth.routing import _is_8b_net, _get_net_length, _get_port_base

class TestRoutingPriority(unittest.TestCase):
    def setUp(self):
        self.nodes_data = {
            'node_8b': {
                'dims': (2, 2, 2),
                'pin_locations': {
                    'data[0]': (0, 0, 0), 'data[1]': (0, 0, 0), 'data[2]': (0, 0, 0), 'data[3]': (0, 0, 0),
                    'data[4]': (0, 0, 0), 'data[5]': (0, 0, 0), 'data[6]': (0, 0, 0), 'data[7]': (0, 0, 0)
                }
            },
            'node_1b': {
                'dims': (2, 2, 2),
                'pin_locations': {
                    'clk': (0, 0, 0), 'rst': (0, 0, 0)
                }
            }
        }
        self.positions = {
            'node_8b': (0, 0, 0),
            'node_1b': (10, 0, 0),
            'sink_near': (2, 0, 0),
            'sink_far': (20, 0, 0)
        }
        # Add sink nodes to nodes_data for completeness
        self.nodes_data['sink_near'] = {'dims': (1,1,1), 'pin_locations': {'in': (0,0,0)}}
        self.nodes_data['sink_far'] = {'dims': (1,1,1), 'pin_locations': {'in': (0,0,0)}}

    def test_is_8b_net(self):
        net_8b = {'driver': ('node_8b', 'data[0]'), 'sinks': [('sink_near', 'in')]}
        net_1b = {'driver': ('node_1b', 'clk'), 'sinks': [('sink_near', 'in')]}
        
        self.assertTrue(_is_8b_net('net_8b', net_8b, self.nodes_data))
        self.assertFalse(_is_8b_net('net_1b', net_1b, self.nodes_data))

    def test_net_length(self):
        net_short = {'driver': ('node_8b', 'data[0]'), 'sinks': [('sink_near', 'in')]}
        net_long = {'driver': ('node_8b', 'data[0]'), 'sinks': [('sink_far', 'in')]}
        
        len_short = _get_net_length(net_short, self.positions, self.nodes_data)
        len_long = _get_net_length(net_long, self.positions, self.nodes_data)
        
        self.assertLess(len_short, len_long)

    def test_sorting_logic(self):
        # Create 4 nets:
        # 1. 8b Short
        # 2. 8b Long
        # 3. 1b Short
        # 4. 1b Long
        
        nets = {
            '8b_short': {'driver': ('node_8b', 'data[0]'), 'sinks': [('sink_near', 'in')]},
            '8b_long': {'driver': ('node_8b', 'data[1]'), 'sinks': [('sink_far', 'in')]},
            '1b_short': {'driver': ('node_1b', 'clk'), 'sinks': [('sink_near', 'in')]},
            '1b_long': {'driver': ('node_1b', 'rst'), 'sinks': [('sink_far', 'in')]}
        }
        
        net_priorities = []
        for net_name, net_data in nets.items():
            is_8b = _is_8b_net(net_name, net_data, self.nodes_data)
            length = _get_net_length(net_data, self.positions, self.nodes_data)
            net_priorities.append((net_name, (not is_8b, length)))
            
        net_priorities.sort(key=lambda x: x[1])
        sorted_nets = [n[0] for n in net_priorities]
        
        expected_order = ['8b_short', '8b_long', '1b_short', '1b_long']
        self.assertEqual(sorted_nets, expected_order)

if __name__ == '__main__':
    unittest.main()

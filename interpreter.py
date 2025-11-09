"""
Interpreter Module — turns Action Sensor logs into a semantic workflow graph.

Takes gui_events.json from the Action Sensor and builds a state graph
showing how users move through UI states and actions.
"""

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import re

from logger import get_logger
log = get_logger("INTERPRETER")


class WorkflowInterpreter:
    """
    Turns raw UI action events into a structured graph of states and transitions.
    Basically: click logs → workflow map.
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_counter = 0
        self.current_state = None
        self.state_history = []

    def load_events(self, filename: str) -> Dict[str, Any]:
        # Load Action Sensor event data from JSON file
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data.get('mouse_events', []))} mouse events")
            print(f"Loaded {len(data.get('keyboard_events', []))} keyboard events")
            return data
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Error: {filename} is not valid JSON")
            return {}

    def create_node_id(self) -> str:
        # Just makes a unique sequential state ID
        self.node_counter += 1
        return f"state_{self.node_counter}"

    def extract_semantic_label(self, event: Dict[str, Any]) -> str:
        # Pull a short, readable label from the event’s description
        action = event.get('action_description', 'Unknown action')
        elem_type = event.get('element', {}).get('type', 'unknown')
        window = event.get('element', {}).get('window', 'unknown')

        # Clean up common phrasing so labels aren’t messy
        label = action
        for prefix in [
            "User clicked ", "User selected ", "User toggled ",
            "User right-clicked to open ", "User switched to ", "User clicked in "
        ]:
            label = label.replace(prefix, "")

        # Remove stray quotes and capitalize it
        label = re.sub(r"['\"]", "", label)
        if label:
            label = label[0].upper() + label[1:]

        return label

    def should_create_node(self, event: Dict[str, Any]) -> bool:
        # Decide if this event is important enough for its own graph node
        elem_type = event.get('element', {}).get('type', '')
        action_desc = event.get('action_description', '')

        # Stuff that usually matters for workflow tracking
        significant_types = [
            'MenuItem', 'Button', 'TabItem',
            'CheckBox', 'RadioButton', 'ListItem'
        ]

        if elem_type in significant_types:
            return True
        if 'window' in elem_type.lower():
            return True
        if 'menu' in action_desc.lower():
            return True

        # Skip boring stuff like generic pane clicks or text edits
        if elem_type in ['Pane', 'Edit', 'Text']:
            return False

        return False

    def classify_node_type(self, event: Dict[str, Any]) -> str:
        # Label what kind of interaction this node represents
        elem_type = event.get('element', {}).get('type', '')
        action = event.get('action_description', '').lower()

        if 'menu' in elem_type.lower() or 'menu' in action:
            return 'menu'
        elif elem_type == 'Button':
            return 'action'
        elif elem_type in ['Edit', 'Text']:
            return 'input'
        elif elem_type in ['TabItem', 'ListItem']:
            return 'selection'
        elif 'window' in elem_type.lower():
            return 'navigation'
        else:
            return 'interaction'

    def build_graph_from_events(self, events_data: Dict[str, Any]) -> None:
        # Main loop — goes through events and builds the graph step by step
        mouse_events = events_data.get('mouse_events', [])
        if not mouse_events:
            # print("No mouse events to process")
            return
        log.debug("Building semantic graph.")
        # print(f"\nBuilding graph from {len(mouse_events)} events...")

        # Start with an initial “App Start” node
        initial_node = {
            'id': self.create_node_id(),
            'label': 'Application Start',
            'type': 'start',
            'timestamp': mouse_events[0]['timestamp']
        }
        self.nodes.append(initial_node)
        self.current_state = initial_node['id']

        for event in mouse_events:
            if not self.should_create_node(event):
                continue  # skip small stuff

            # Build a new node for this event
            node = {
                'id': self.create_node_id(),
                'label': self.extract_semantic_label(event),
                'type': self.classify_node_type(event),
                'timestamp': event['timestamp'],
                'element_type': event.get('element', {}).get('type'),
                'window': event.get('element', {}).get('window'),
                'position': event.get('position'),
                'patterns': event.get('patterns', {})
            }
            self.nodes.append(node)

            # Link it to the previous node
            if self.current_state:
                edge = {
                    'id': f"edge_{len(self.edges) + 1}",
                    'from': self.current_state,
                    'to': node['id'],
                    'action': event.get('action_description'),
                    'timestamp': event['timestamp']
                }

                # Rough transition timing (in ms)
                prev_node = next((n for n in self.nodes if n['id'] == self.current_state), None)
                if prev_node and prev_node.get('timestamp'):
                    try:
                        t1 = datetime.fromisoformat(prev_node['timestamp'])
                        t2 = datetime.fromisoformat(event['timestamp'])
                        edge['duration_ms'] = int((t2 - t1).total_seconds() * 1000)
                    except:
                        pass

                self.edges.append(edge)

            self.current_state = node['id']
            self.state_history.append(node['id'])

        # print(f"Created {len(self.nodes)} nodes and {len(self.edges)} edges")

    def detect_workflows(self) -> List[Dict[str, Any]]:
        # Look for simple repeated patterns of user workflows
        workflows = []
        current_workflow = []

        for node in self.nodes:
            if node['type'] == 'menu':
                # Start new chain whenever a menu shows up
                if current_workflow:
                    workflows.append({
                        'id': f"workflow_{len(workflows) + 1}",
                        'states': current_workflow.copy(),
                        'start': current_workflow[0],
                        'end': current_workflow[-1]
                    })
                current_workflow = [node['id']]
            else:
                current_workflow.append(node['id'])

        # Wrap up final sequence
        if current_workflow:
            workflows.append({
                'id': f"workflow_{len(workflows) + 1}",
                'states': current_workflow.copy(),
                'start': current_workflow[0],
                'end': current_workflow[-1]
            })

        return workflows

    def calculate_statistics(self) -> Dict[str, Any]:
        # Basic stats about the graph (counts, timing, etc.)
        node_types = defaultdict(int)
        for node in self.nodes:
            node_types[node.get('type', 'unknown')] += 1

        total_duration = sum(edge.get('duration_ms', 0) for edge in self.edges)

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'total_duration_ms': total_duration,
            'avg_transition_time_ms': total_duration / len(self.edges) if self.edges else 0
        }

    def export_graph(self, filename: str = 'workflow_graph.json') -> None:
        # Dump graph + workflow data to JSON file
        workflows = self.detect_workflows()
        stats = self.calculate_statistics()

        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source': 'WorkflowInterpreter',
                'statistics': stats
            },
            'nodes': self.nodes,
            'edges': self.edges,
            'workflows': workflows
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # print(f"\nGraph exported to {filename}")
        # print(f"Stats:")
        # print(f"  Nodes: {stats['total_nodes']}")
        # print(f"  Edges: {stats['total_edges']}")
        # print(f"  Workflows: {len(workflows)}")
        # print(f"  Node types: {stats['node_types']}")
        # print(f"  Total duration: {stats['total_duration_ms']/1000:.2f}s")

    def print_graph_summary(self) -> None:
        # Quick readable summary of what got built
        print("\n" + "=" * 60)
        print("WORKFLOW GRAPH SUMMARY")
        print("=" * 60)

        print(f"\nStates (Nodes): {len(self.nodes)}")
        for i, node in enumerate(self.nodes[:10]):  # only first 10
            print(f"  {i+1}. [{node['type']}] {node['label']}")
        if len(self.nodes) > 10:
            print(f"  ...and {len(self.nodes) - 10} more")

        print(f"\nTransitions (Edges): {len(self.edges)}")
        for i, edge in enumerate(self.edges[:10]):
            from_node = next((n for n in self.nodes if n['id'] == edge['from']), None)
            to_node = next((n for n in self.nodes if n['id'] == edge['to']), None)
            if from_node and to_node:
                dur = f" ({edge.get('duration_ms', 0)}ms)" if 'duration_ms' in edge else ""
                print(f"  {i+1}. {from_node['label']} → {to_node['label']}{dur}")
        if len(self.edges) > 10:
            print(f"  ...and {len(self.edges) - 10} more")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Interpreter — build semantic graphs from Action Sensor logs"
    )
    parser.add_argument('--input', '-i', default='gui_events.json',
                        help='Input Action Sensor JSON (default: gui_events.json)')
    parser.add_argument('--output', '-o', default='workflow_graph.json',
                        help='Output graph file (default: workflow_graph.json)')
    parser.add_argument('--summary', action='store_true',
                        help='Print a readable summary to console')

    args = parser.parse_args()

    interpreter = WorkflowInterpreter()
    events_data = interpreter.load_events(args.input)

    if not events_data:
        print("No events loaded. Exiting.")
        return

    interpreter.build_graph_from_events(events_data)
    interpreter.export_graph(args.output)

    if args.summary:
        interpreter.print_graph_summary()


if __name__ == '__main__':
    main()

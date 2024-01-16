import os

import tensorflow as tf
import keras
from keras.src.engine import functional

from collections import defaultdict
from typing import List
from collections.abc import Sized

def load_converted_tf_network(network_name: str) -> keras.Model:
    """
    Load the converted network from 
    """

    # Load the weights
    network_path = os.path.join('models', 'converted-tf', f'{network_name}.keras')

    return keras.models.load_model(network_path)



def explore_topology(model):
    adjacency_list = defaultdict(list)
    submodels_first_layers = {}
    submodels_last_layers = {}

    for i, curr_layer in enumerate(model.layers):
        print(f'Current layer: {curr_layer.name}')
        if isinstance(curr_layer, functional.Functional):
            submodels_first_layers[curr_layer] = curr_layer.layers[0]
            submodels_last_layers[curr_layer] = curr_layer.layers[-1]
        

    for i, curr_layer in enumerate(model.layers):
        if isinstance(curr_layer, functional.Functional):
            sub_adj_list = explore_topology(curr_layer)
            adjacency_list |= sub_adj_list
        for j, node in enumerate(curr_layer._inbound_nodes):
            node_key = curr_layer.name + "_ib-" + str(j)
            if node_key in model._network_nodes:
                for inbound_layer in tf.nest.flatten(node.inbound_layers):
                    if isinstance(curr_layer, functional.Functional):
                        actual_curr_layer = submodels_first_layers[curr_layer]
                    else:
                        actual_curr_layer = curr_layer
                    
                    if isinstance(inbound_layer, functional.Functional):
                        actual_inbound_layer = submodels_last_layers[inbound_layer]
                    else:
                        actual_inbound_layer = inbound_layer
                    adjacency_list[actual_inbound_layer].append(actual_curr_layer)

    return dict(adjacency_list)


def reverse_adj_list(adj_list):
    rev_adj_list = defaultdict(list)
    for edge_from, edges_to_list in adj_list.items():
        for edge_to in edges_to_list:
            rev_adj_list[edge_to].append(edge_from)
    return dict(rev_adj_list)

def print_graph(adj_list, reverse=False):
    for target_layer, bound_layers in adj_list.items():
        bound_names = [x.name for x in bound_layers]
        if reverse:
            print(f'{target_layer.name} <-- {bound_names} ')
        else:
            print(f'{bound_names} --> {target_layer.name}')


def toposort_graph(adj_list):
    def dfs_internal(node):
        temp_marked = set()
        if node not in nodes_not_perm_marked:
            return
        if node in temp_marked:
            raise ValueError(f'Circular Dependency in {node}')
        
        temp_marked.add(node)

        for adj_node in adj_list.get(node, []):
            dfs_internal(adj_node)
        
        temp_marked.remove(node)
        nodes_not_perm_marked.remove(node)
        sorted_adj_list[node] = adj_list[node]

    sorted_adj_list = {}
    nodes_not_perm_marked = set(adj_list.keys())
    while len(nodes_not_perm_marked) > 0:
        node_to_visit = next(iter(nodes_not_perm_marked))
        dfs_internal(node_to_visit)
    return dict(reversed(sorted_adj_list.items()))
    
    
def remap_adj_list_keys_to_str(adj_list):
    remapped_adj_list = {}
    for edge_from, edges_to_list in adj_list.items():
        remapped_adj_list[edge_from.name] = edges_to_list
    return remapped_adj_list

def clone_model(model, input_layers, output_layers, copy_weights = True, verbose=False):

    toposorted_adj_list = toposort_graph(explore_topology(model))
    rev_adj_list = reverse_adj_list(toposorted_adj_list)

    if not isinstance(input_layers, Sized):
        input_layers = [input_layers]

    if not isinstance(output_layers, Sized):
        output_layers = [output_layers]

    # Clone the inputs in order to create a separate graph
    cloned_inputs = {inp: keras.Input(inp._batch_input_shape[1:], name = f'{inp.name}_new') for inp in input_layers}
    intermediate_tensors = dict(cloned_inputs)

    # Iterate nodes in topological order
    for i, curr_layer in enumerate(toposorted_adj_list):
        if curr_layer in cloned_inputs:
            # Model inputs have no inbound layers
            continue

        # List of operands of the current layers (layers connected inbound to curr_layer)
        inbound_nodes_list = rev_adj_list[curr_layer]
        # Input tensors are guaranteed to be present because we are walking the graph in topological order
        layer_inputs = [intermediate_tensors[inbound_node] for inbound_node in inbound_nodes_list]


        # Additional kwargs are stored in the inbound node
        call_kwargs : dict = curr_layer._inbound_nodes[0].call_kwargs
        # TODO: Tricky workaround to remove duplicated arguments for binary operators
        if len(inbound_nodes_list) == 2:
            del call_kwargs['y']
        if verbose:
            print(f'{i}. Building layer: {curr_layer.name} ({curr_layer})')
            print(f'{i}. Operands: {[inp.name for inp in layer_inputs]} ({layer_inputs})')
            print(f'{i}. Additional kwargs: {call_kwargs}')
        # Call the layer to build the graph
        # Put the intermediate tensors coming from the inbound layers as positional
        # and additional configuration kwargs
        intermediate_tensors[curr_layer] = curr_layer(*layer_inputs, **call_kwargs)

    outputs = [res for layer, res in intermediate_tensors.items() if layer in output_layers]

    if verbose:
        print('---')
        print('Intermediate Results:')
        for layer, tensor in intermediate_tensors.items():
            print(f'{layer.name}: {tensor}')
        print('---')
        print('Assembled model. Now calling keras.Model constructor with the following kwargs')
        print(f'Inputs: {list(cloned_inputs.values())}')
        print(f'Outputs: {outputs}')
    cloned_model = keras.Model(inputs=list(cloned_inputs.values()), outputs=outputs) 
    if copy_weights:
        cloned_model.set_weights(model.get_weights())
    return cloned_model
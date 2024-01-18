import os
import tensorflow as tf
import keras
from keras.src.engine import functional
from keras.src.layers import Layer
from keras.src.engine.keras_tensor import KerasTensor

from collections import defaultdict
from typing import Callable, Iterable, List, Dict, Union, Optional
from collections.abc import Sized


def load_converted_tf_network(network_name: str) -> keras.Model:
    """
    Load the converted network from
    """

    # Load the weights
    network_path = os.path.join("models", "converted-tf", f"{network_name}.keras")

    return keras.models.load_model(network_path)


def explore_topology(model: functional.Functional) -> Dict[Layer, List[Layer]]:
    """
    Explores the topology model and its topology with a DFS search, reconstucting
    the graph of the model

    Args:
        model: A Keras Functional model

    Returns:
        Returns the adjacency list stored as a dict. Each layer is mapped to a list of layers directly depending on it.
    """
    adjacency_list = defaultdict(list)
    submodels_first_layers = {}
    submodels_last_layers = {}

    for i, curr_layer in enumerate(model.layers):
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


def reverse_adj_list(adj_list: Dict[Layer, List[Layer]]) -> Dict[Layer, List[Layer]]:
    rev_adj_list = defaultdict(list)
    for edge_from, edges_to_list in adj_list.items():
        for edge_to in edges_to_list:
            rev_adj_list[edge_to].append(edge_from)
    return dict(rev_adj_list)


def print_graph(adj_list: Dict[Layer, List[Layer]], reverse=False):
    """
    Prints the graph to stdout

    Args:
        adj_list: An adjacency list of Layers, representing a model

    Returns:
        None
    """
    for target_layer, bound_layers in adj_list.items():
        bound_names = [x.name for x in bound_layers]
        if reverse:
            print(f"{target_layer.name} <-- {bound_names} ")
        else:
            print(f"{bound_names} --> {target_layer.name}")


def toposort_graph(adj_list: Dict[Layer, List[Layer]]) -> Dict[Layer, List[Layer]]:
    """
    Sort a graph of a model, represented as an adjacency list, following the topological order.

    Args:
        adj_list: The model graph to sort topologically represented by an adjacency list of keras Layers.

    Returns:
        Returns an ordered dictionary, with the same semantics of the input adjacency list, but sorted by topological order of the keras Layers.
    """
    # The algorithm uses the DFS strategy, starting from a random node and fiding all the nodes of the DAG that have no non-marked dependants
       
    # Internal dfs algorithm that discovers the terminal nodes and marks them 
    def dfs_internal(node: Layer):
        temp_marked = set()
        if node not in nodes_not_perm_marked:
            # If marked do, nothing
            return
        if node in temp_marked:
            raise ValueError(f"Circular Dependency in {node}")

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
    # The obtained adjacency list is reversed since the terminal nodes are inserted in the list first, so it needs to
    # be reversed.
    return dict(reversed(sorted_adj_list.items()))


def remap_adj_list_keys_to_str(adj_list : Dict[Layer, List[Layer]]) -> Dict[str, List[Layer]]:
    # DEBUG METHOD for accessing nodes by their names
    remapped_adj_list = {}
    for edge_from, edges_to_list in adj_list.items():
        remapped_adj_list[edge_from.name] = edges_to_list
    return remapped_adj_list


def clone_model(
    model : keras.Model,
    input_layers : Union[Iterable[Layer], Layer],
    output_layers : Union[Iterable[Layer], Layer, None] = None,
    layer_factory : Callable[[Layer, Iterable[KerasTensor], KerasTensor], Optional[Layer]] = lambda layer, inputs: keras.layers.Identity(),
    copy_weights=True,
    verbose=False,
) -> keras.Model:
    """
    Clones a keras model giving the option to insert new layers in the middle of the model.
    Allows the user to add custom layers in the new cloned model

    Args:
        model: The keras model that will be cloned.

        input_layers: The input layer or a list of layers containing all the input layers of the original model.

        output_layers: The output layer or a list of layers containing all the output layers of the original model. 
            Note that in keras the output of a model is not layer but the tensors returned by that layers. Differently than keras, 
            this argument must contain the layers that direclty produce those output tensors.

        layer_factory: A callback that is used to decide where and which new layers add to the model.
            The callback is called before cloning each layer, and receives two parameters, the layer object that will is currently being cloned an
            the input tensors of that layer. If the callback returns a layer, that layer will be added after the layer currently being cloned.
            If the callback returns None, no new layer will be added in this occasion. If no callback is specified, no layers will be added.

        copy_weights: If true the weights of the old model, will be copied to the cloned model. If layers with weights are added to the cloned model,
            the copy will fail. Defaults to True.

        verbose: Enable additional logging to stdout. Defaults to False.
        

    Returns:
        Returns a cloned model of the original one with eventual new layers added as specfied, and if desired with old model's weights copied in the new one.
    """
    toposorted_adj_list = toposort_graph(explore_topology(model))
    rev_adj_list = reverse_adj_list(toposorted_adj_list)

    if not isinstance(input_layers, Sized):
        input_layers = [input_layers]

    if output_layers is not None and not isinstance(output_layers, Sized):
        output_layers = [output_layers]

    # Clone the inputs in order to create a separate graph
    cloned_inputs = {
        inp: keras.Input(inp._batch_input_shape[1:], name=f"{inp.name}_new")
        for inp in input_layers
    }
    intermediate_tensors = dict(cloned_inputs)

    # Iterate nodes in topological order
    for i, curr_layer in enumerate(toposorted_adj_list):
        if curr_layer in cloned_inputs:
            # Model inputs have no inbound layers
            continue

        # List of operands of the current layers (layers connected inbound to curr_layer)
        inbound_nodes_list = rev_adj_list[curr_layer]
        # Input tensors are guaranteed to be present because we are walking the graph in topological order
        layer_inputs = [
            intermediate_tensors[inbound_node] for inbound_node in inbound_nodes_list
        ]

        # Additional kwargs are stored in the inbound node
        call_kwargs: dict = curr_layer._inbound_nodes[0].call_kwargs
        # TODO: Tricky workaround to remove duplicated arguments for binary operators
        if len(inbound_nodes_list) == 2:
            if "y" in call_kwargs:
                del call_kwargs["y"]
            if "shape" in call_kwargs:
                del call_kwargs["shape"]
        if verbose:
            print(f"{i}. Layer: ({curr_layer})")
            print(f"\t* Name: {curr_layer.name}")
            print(
                f'\t* Operands ({len(layer_inputs)}): '
            )
            for inp in layer_inputs:
                if hasattr(inp, "name"):
                    print(f'\t\t- {inp}')
                else:
                    print(f'\t\t- {inp}')
            print(f"\t* Additional kwargs: {call_kwargs}")
            print("")
        # Call the layer to build the graph
        # Put the intermediate tensors coming from the inbound layers as positional
        # and additional configuration kwargs
        layer_output = curr_layer(*layer_inputs, **call_kwargs)

        layer_to_inject = layer_factory(curr_layer, layer_inputs, layer_output)
        intermediate_tensors[curr_layer] = layer_output
        if layer_to_inject is not None:
            if verbose:
                print(
                    f"\t* INJECTED LAYER: {layer_to_inject}"
                )
            intermediate_tensors[curr_layer] = layer_to_inject(
                layer_output
            )
        else:
            intermediate_tensors[curr_layer] = layer_output

    if output_layers is None:
        output_layers = [curr_layer]

    outputs = [
        res for layer, res in intermediate_tensors.items() if layer in output_layers
    ]

    if verbose:
        print("---")
        print(
            "Assembled model. Now calling keras.Model constructor with the following kwargs"
        )
        print(f"Inputs: {list(cloned_inputs.values())}")
        print(f"Outputs: {outputs}")
    cloned_model = keras.Model(inputs=list(cloned_inputs.values()), outputs=outputs)
    if copy_weights:
        cloned_model.set_weights(model.get_weights())
    return cloned_model

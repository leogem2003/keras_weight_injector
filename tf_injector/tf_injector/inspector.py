from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import logging
import inspect

logging.basicConfig(level=logging.DEBUG)


class Inspector:
    def __init__(self, model: Model):
        self.outputs: list[str] = []
        self._model = self.inspect_naive(model)

    def __call__(self, data):
        mapping = []
        for layer, tensor in zip(self.outputs, self._model(data)):
            if isinstance(tensor, list):
                tensor = [t.numpy() for t in tensor]
            else:
                tensor = [tensor.numpy()]
            mapping.append((layer, tensor))
        return mapping

    def compare(self, d1, d2, func):
        return [
            (layer, func(v1, v2))
            for (layer, v1), (_, v2) in zip(self.__call__(d1), self.__call__(d2))
        ]

    def inspect(self, model: Model) -> Model:
        inserted_tensors: dict[str, tf.Tensor] = {}

        new_outputs = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            print("Inserting", layer.name)
            # rebuild the layer with updated inputs
            layer_inputs = []
            orig_inputs = []
            for node in layer._inbound_nodes:
                for (
                    inbound_layer,
                    node_index,
                    output_index,
                    tensor,
                ) in node.iterate_inbound():
                    if not isinstance(
                        inbound_layer, tf.keras.layers.InputLayer
                    ):  # connection to an Input: no middleware
                        ins_tensor = inserted_tensors[inbound_layer.name]
                        print(
                            f"Found input tensor from {inbound_layer.name}, connecting with middleware {ins_tensor}"
                        )
                    else:
                        ins_tensor = tensor

                    if isinstance(ins_tensor, list):
                        ins_tensor = ins_tensor[output_index]

                    layer_inputs.append(ins_tensor)
                    orig_inputs.append(tensor)

            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            try:
                layer_output = layer(layer_inputs)
            except TypeError as e:  # reshape hack
                print(orig_inputs)
                print(inspect.signature(layer.call))
                print(layer.get_config())
                print(model.get_config())
                layer_output = layer(orig_inputs[0], orig_inputs[1])
            finally:
                # insert the middleware in the output
                middleware = tf.keras.layers.Identity(name=f"mid_{layer.name}")(
                    layer_output
                )
                new_outputs.append(middleware)
                inserted_tensors[layer.name] = middleware

        return Model(inputs=model.input, outputs=new_outputs)

    def inspect_naive(self, model: Model) -> Model:
        outputs = []
        inspected = None
        for layer in model.layers:
            outputs.append(layer.output)
            try:
                inspected = Model(inputs=model.input, outputs=outputs)
                self.outputs.append(layer.name)
                print("Added", layer.name)
            except:
                print("Can't add", layer.name)
                outputs.pop()

        if inspected is None:
            raise RuntimeError
        return inspected


def diff_perc(f, s):
    return (f != s).sum() / f.size


def diff_dist(f, s):
    return np.sqrt(((f - s) ** 2).sum())


def diff_max(f, s):
    return np.abs(f - s).max()


def diff_avg(f, s):
    return np.average(np.abs(f - s))


def diff_composite(f, s):
    res = []
    for i1, i2 in zip(f, s):
        res.append(
            (
                diff_perc(i1, i2),
                diff_dist(i1, i2),
                diff_max(i1, i2),
                diff_avg(i1, i2),
            )
        )
    return res

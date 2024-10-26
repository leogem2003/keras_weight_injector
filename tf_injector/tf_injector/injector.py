import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore
from tqdm import tqdm  # type:ignore
import numpy as np
import csv
import shutil
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import TypeAlias
from tf_injector.utils import INJECTED_LAYERS_TYPES

FaultType: TypeAlias = tuple[str, tuple[int, ...], int]


@dataclass
class FaultList:
    # [("layer", (coords,..), bitpos), ...]
    faults: list[FaultType] = field(default_factory=lambda: [])
    resume_idx: int = 0


class Injector:
    """
    class Injector
    performs fault injections on tensorflow networks
    """

    def __init__(
        self, network: keras.Model, dataset: tf.data.Dataset, sort_layers: bool = False
    ):
        """
        Args:
            network: the target network
            dataset: the dataset on which the inferences are run
            sort_layers: (NAT)sort layers
        """
        self.network = network
        self.dataset = dataset
        self.target_layers: dict[str, keras.Layer] = {
            layer.name: layer
            for layer in network._flatten_layers(
                include_self=False, recursive=True
            )  # extracts all layers
            if isinstance(layer, INJECTED_LAYERS_TYPES)
        }

        self.sort_layers = sort_layers
        self.faults = FaultList()

    def load_fault_list(self, fault_path: str, resume_from: int = 0):
        """
        Loads a fault list from a csv file and runs compatibility checks with the network
        Args:
            fault_path: filename of the fault list, must be a csv file
            resume_from: only saves injections starting from its value (default=0)
        """
        self._reset_fault()
        included_layers = set()
        self.faults.resume_idx = resume_from
        with open(fault_path, "r") as f:
            reader = csv.reader(f)
            next(iter(reader))  # skip header
            for row in reader:
                if len(row) != 4:
                    raise ValueError(
                        f"invalid row format: expected 4 columns, got {len(row)}, on row {row}"
                    )
                id, layer, coords, bit = row
                included_layers.add(layer)

                # get coords as a tuple of ints
                int_coords = tuple((int(coord) for coord in coords[1:-1].split(",")))
                # TODO: check coords correctness

                if int(id) >= resume_from:
                    self.faults.faults.append((layer, int_coords, int(bit)))

        # all target layers are in injection list
        assert set(self.target_layers.keys()) == included_layers

    @staticmethod
    def _tqdm(iterable, faulty: bool) -> tqdm:
        return tqdm(
            iterable,
            colour="red" if faulty else "green",
            desc="Faulty Run" if faulty else "Clean Run",
            ncols=shutil.get_terminal_size().columns,
        )

    def _run_inference_on_batch(self, data):
        return self.network(data).numpy()

    def run_inference(self, batch: int, faulty: bool = False):
        batched = self.dataset.batch(batch)
        pbar = self._tqdm(batched, faulty)
        for batch in pbar:
            # type:ignore
            data, label = batch
            self._run_inference_on_batch(data)

    def run_campaign(self, batch: int = 64):
        """
        Runs a campaign with the loaded fault list
        Params:
            batch: inference batch size (default=64)
        """
        if not self.faults.faults:
            raise RuntimeError(
                "attempting to run a campaign without a fault list loaded"
            )

        pbar = self._tqdm(self.faults.faults, False)
        fault_id = self.faults.resume_idx
        for fault in pbar:
            with self._apply_fault(fault):
                self.run_inference(batch, faulty=True)
            fault_id += 1

    def _reset_fault(self):
        self.faults.faults.clear()
        self.faults.resume_idx = 0

    @contextmanager
    def _apply_fault(self, fault: FaultType):
        target_layer_name, coords, bitpos = fault
        target_layer = self.target_layers[target_layer_name]
        bitmask = np.uint32(1) << bitpos
        layer_weights = target_layer.get_weights()
        target_weights = layer_weights[0].view(dtype=np.uint32)
        # clean_value = target_weights[coords] # for assert test
        try:
            target_weights[coords] ^= bitmask
            target_layer.set_weights(layer_weights)
            # assert (target_weights[coords] ^ clean_value) == bitmask
            yield target_weights
        finally:
            target_weights[coords] ^= bitmask
            # assert target_weights[coords] == clean_value
            target_layer.set_weights(layer_weights)

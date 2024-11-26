# PT x TF utils
ptxtf_utils is a library of utility scripts for converting fault lists between Torch and TensorFlow.
Each module is runnable as a standalone python script.

## Requirements
ptxtf_net.py requires any version of Torch and any version of TensorFlow2. Run `pip install -r requirements.txt` to ensure you have all the requirements installed. 
ptxtf_fault.py does not have any requirements.

## Modules
- `ptxtf_fault.py`: converts a fault list between PT and TF (both directions aviable). Requires an input fault list correctly formatted and a file containing the match between layers, in the format outputted by `ptxtf_net.py`. Run `python ptxtf_fault.py --help` for more information.
Expected input format for the fault list:
| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |
> [!NOTE]
> This script only converts the layer name and permutes TensorIndex, it does not check the correctness of the fault.
- `ptxtf_net.py`: outputs a matching of the layers of corresponding networks. For the Torch network, a python file containing its definition is requires, whereas for Tensorflow a .keras file is needed.  Run `python ptxtf_net.py --help` for more information.
Output format:
```
pt_layer0,tf_layer0
pt_layer1,tf_layer1
...
pt_layern, tf_layern
```
> [!WARNING]
> Please be sure of the safety of the Torch script, since it will be entirely executed.

> [!NOTE]
> At the moment, only the layers targeted by `tf_injector` are supported:
> - Torch: Conv2d, Linear 
> - TensorFlow: keras.layers.Conv2D, keras.layers.Dense

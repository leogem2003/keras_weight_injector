# PT x TF utils
ptxtf_utils is a library of utility scripts for converting fault lists between Torch and TensorFlow.
Each module is runnable as a standalone python script.

## Requirements
ptxtf_net.py requires any version of Torch and any version of TensorFlow2. Run `pip install -r requirements.txt` to ensure you have all the requirements installed. 
ptxtf_fault.py does not have any requirements.

## Modules
- `ptxtf_fault.py`: converts a fault list between PT and TF (both directions aviable). Run `python ptxtf_fault.py --help` for more information.
- `ptxtf_net.py`: outputs a matching of the layers of corresponding networks. For the Torch network, a python file containing its definition is requires, whereas for Tensorflow, a .keras file is needed. Run `python ptxtf_net.py --help` for more information.


# Dataset converter
Converts Torch datasets into (compressed) Tensorflow datasets.
Datasets are downloaded through torchvision and they are saved with the relative transformations applied.
The resulting dataset is ready-to-use with tensorflow models.

## Installation
Requires Python 3.10

Create virtual environment
`python -m venv venv_name && venv_name/bin/activate`

Install dependencies
`pip install -r requirements.txt`


## Usage
Run as Python module
`python -m dataset_converter ARGS, ...`

For all informations and options, run
`python -m dataset_converter --help`

## Output
Tensorflow dataset.
To use this dataset, run
```
import tensorflow as tf
tf.data.load("path/to/dataset", compression="GZIP")
```

## Supported datasets
- CIFAR10
- CIFAR100
- GTSRB 

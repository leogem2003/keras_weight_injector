# Deep Neural Network Models for Reliability Studies
Welcome to the repository containing state-of-the-art Deep Neural Network (DNN) models implemented in both PyTorch and TensorFlow for conducting reliability studies. 

## Project Collaboration

This project is a collaboration between the following institutions:

- [Politecnico di Torino](https://www.polito.it/)
- [Politecnico di Milano](https://www.polimi.it/)
- [Ecole Centrale de Lyon](https://www.ec-lyon.fr/en)

## Getting Started

A clean Pytorch inference can be executed with the following programm:
```
python main.py -n network-name -b batch-size 
```

It is possible to execute inferences with available GPUs sepcifing the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/network_name/pt``` folder. 

## Available Models (so far)

The TensorFlow versions of the models, when available, are converted using the [nobuco](https://github.com/AlexanderLutsenko/nobuco) PyTorch to Tensorflow converter, and have the same accuracy of the PyTorch models.

### ResNet Models

ResNet models are trained on CIFAR-10. 


| Model        | Description                   | PyTorch                 | Tensorflow         | Accuracy |
| ------------ | ----------------------------- | ----------------------- |--------------------|----------|
| ResNet20     | 20-layer Residual Network     | :white_check_mark:      | :white_check_mark: | 91.48 %  |
| ResNet32     | 32-layer Residual Network     | :white_check_mark:      | :white_check_mark: | 92.34 %  |
| ResNet44     | 44-layer Residual Network     | :white_check_mark:      | :white_check_mark: | 92.76 %  |
| ResNet56     | 56-layer Residual Network     | :white_check_mark:      | :white_check_mark: | 93.27 %  | 
| ResNet110    | 110-layer Residual Network    | :white_check_mark:      | :white_check_mark: | 93.5  %  |
| ResNet1202   | 1202-layer Residual Network   | :white_check_mark:      | :white_check_mark: | 93.5  %  |


## Prerequisites 

Before running inferences, install the needed packages and tools
```
pip install -r requirements.txt
```


## Analyse the results: from .pt to .csv

Results file can be converted to csv using the script:
```
python pt_to_csv.py -n network-name -b batch-size 
```
Results are saved in the ```output/network_name/csv``` folder. Notice that carrying out operation on the CSV file is going to be more expensive than carrying out the same analysis on .pt files. This format should be used only for data visualization purposes only.


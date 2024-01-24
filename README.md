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

The Keras versions of the models, when available, are obtained using the [nobuco](https://github.com/AlexanderLutsenko/nobuco) PyTorch to Keras converter.
The Keras versions of all models share the same structure and weigths, and have similar accuracies to their PyTorch counterpart.

### CIFAR-10 Models

All the models are validated using the CIFAR10 validation tests, that cointains 10000 images.

| Model        | Description                   | PyTorch  Accuracy       | Keras Accuracy     |  Sources  |
| ------------ | ----------------------------- | ----------------------- |--------------------|-----------|
| ResNet20     | 20-layer Residual Network     | 91.5 %                  | 91.5 %             |           |
| ResNet32     | 32-layer Residual Network     | 92.3 %                  | 92.3 %             |           |
| ResNet44     | 44-layer Residual Network     | 92.8 %                  | 92.8 %             |           |
| ResNet56     | 56-layer Residual Network     | 93.3 %                  | 93.3 %             |           |
| ResNet110    | 110-layer Residual Network    | 93.5 %                  | 93.5 %             |           |
| MobileNetV2  |                               | 91.7 %                  | 91.7 %             |           |
| Vgg19_bn     |                               | 93.2 %                  | 93.2 %             |           | 
| Vgg16_bn     |                               | 93.5 %                  | 93.5 %             |           | 
| Vgg13_bn     |                               | 93.8 %                  | 93.8 %             |           | 
| Vgg11_bn     |                               | 91.3 %                  | 91.3 %             |           |
| DenseNet121  |                               | 93.2 %                  | 91.3 %             |           | 
| DenseNet161  |                               | 93.1 %                  | 93.3 %             |           |          
| DenseNet169  |                               | 93.4 %                  | 92.9 %             |           | 
| GoogLeNet    |                               | 92.2 %                  | 91.4 %             |           | 

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


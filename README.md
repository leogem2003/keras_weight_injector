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
Here is a list of models trained for CIFAR10 dataset, that has images belonging to 10 classes.
All the models are validated using the CIFAR10 validation set, that cointains 10000 images.

| Model        | PyTorch TOP-1 Accuracy  | Keras TOP-1 Accuracy |  Sources  | Notes     |
| ------------ | ----------------------- | -------------------- |---------- |-----------|
| ResNet20     | 91.5 %                  | 91.5 %               |           |           |
| ResNet32     | 92.3 %                  | 92.3 %               |           |           |
| ResNet44     | 92.8 %                  | 92.8 %               |           |           |
| ResNet56     | 93.3 %                  | 93.3 %               |           |           |
| ResNet110    | 93.5 %                  | 93.5 %               |           |           |
| MobileNetV2  | 91.7 %                  | 91.7 %               |           |           |
| Vgg19_bn     | 93.2 %                  | 93.2 %               |           |           |
| Vgg16_bn     | 93.5 %                  | 93.5 %               |           |           |
| Vgg13_bn     | 93.8 %                  | 93.8 %               |           |           | 
| Vgg11_bn     | 91.3 %                  | 91.3 %               |           |           |
| DenseNet121  | 93.2 %                  | 91.3 %               |           |           | 
| DenseNet161  | 93.1 %                  | 93.3 %               |           |           |          
| DenseNet169  | 93.4 %                  | 92.9 %               |           |           | 
| GoogLeNet    | 92.2 %                  | 91.4 %               |           |           | 

### CIFAR-100 Models
Here is a list of models trained for CIFAR100 dataset, that has images belonging to 100 classes.
All the models are validated using the CIFAR100 validation set, that cointains 10000 images.

| Model        | PyTorch TOP-1 Accuracy  | Keras TOP-1 Accuracy |  Sources  | Notes                     |
| ------------ | ----------------------- |--------------------- |---------- | ------------------------- |
| ResNet18     | 76.2 %                  | 76.2 %               |           |                           |
| DenseNet121  | 78.7 %                  | 78.7 %               |           |                           |
| GoogLeNet    | 76.3 %                  | 72.9 %   :warning:   |           | (Imprecise MaxPooling conversion)                               |

### GTSRB Models
Here is a list of models trained for GTSRB dataset, containing 43 classes of German Traffic signals.
All the models are validated using the GTSRB validation set, that cointains 12640 images.

| Model        | PyTorch TOP-1 Accuracy  | Keras TOP-1 Accuracy |  Sources  | Notes                     |
| ------------ | ----------------------- |--------------------- |---------- | ------------------------- |
| ResNet20     |                         |                      |           | (Conversion Failed)       |
| DenseNet121  | 96.5%                   | 96.5%                |           |                           |
| Vgg11_bn     |                         |                      |           | (Conversion Failed)       |

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


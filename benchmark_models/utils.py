import os
import argparse

import torch
from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets
from models.utils import load_from_dict

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    densenet121,
    DenseNet121_Weights,
)


from models.cifar10.densenet import densenet121, densenet161, densenet169
from models.cifar10.googlenet import GoogLeNet
from models.cifar10.mobilenetv2 import MobileNetV2
from models.cifar10.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.cifar10.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.cifar10.inception import Inception3
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader

class UnknownNetworkException(Exception):
    pass


SUPPORTED_MODELS_LIST = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121', 'DenseNet161', 'DenseNet169', 'GoogLeNet', 'MobileNetV2', 
                                 'InceptionV3', 'Vgg11_bn', 'Vgg13_bn', 'Vgg16_bn', 'Vgg19_bn']

def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Run Inferences',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')
    parser.add_argument('--network-name', '-n', type=str,
                        required=True,
                        help='Target network',
                        choices=SUPPORTED_MODELS_LIST)
    parser.add_argument('--tensorflow', '--tf', help='Execute the network in TensorFlow. Convert it a conversion does not exists', action='store_true')
    parsed_args = parser.parse_args()

    return parsed_args

def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               network: torch.nn.Module = None, permute_tf=False) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if 'ResNet' or 'MobileNetV2' or 'GoogLeNet' or 'DenseNet' or 'MobileNet' or 'Inception' or 'Vgg' in network_name and network_name not in ['ResNet18', 'ResNet50']:
        print('Loading CIFAR10 dataset')
        train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class, permute_tf=permute_tf)
        
    # elif 'MobileNetV2' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)
    # elif 'GoogLeNet' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)
    
    # elif 'DenseNet' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)
    # elif 'MobileNet' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)
    # elif 'Inception' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)
    # elif 'Vgg' in network_name:
    #     train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
    #                                          test_image_per_class=image_per_class)    
    # # elif 'LeNet' in network_name:
    #     print('Loading MNIST dataset')
    #     train_loader, _, loader = load_MNIST_datasets(test_batch_size=batch_size)
    else:
        if image_per_class is None:
            image_per_class = 5
        loader = load_ImageNet_validation_set(batch_size=batch_size,
                                              image_per_class=image_per_class,
                                              network=network,)
        train_loader = load_ImageNet_validation_set(batch_size=batch_size,
                                                   image_per_class=None,
                                                   network=network,)

    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return train_loader, loader

def load_network(network_name: str, device: torch.device) -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :return: The loaded network
    """

    if "ResNet" in network_name:
        if network_name == "ResNet20":
            network_function = resnet20
        elif network_name == "ResNet32":
            network_function = resnet32
        elif network_name == "ResNet44":
            network_function = resnet44
        elif network_name == "ResNet56":
            network_function = resnet56
        elif network_name == "ResNet110":
            network_function = resnet110
        elif network_name == "ResNet1202":
            network_function = resnet1202
        else:
            raise UnknownNetworkException(
                f"ERROR: unknown version of ResNet: {network_name}"
            )

        # Instantiate the network
        network = network_function()

        # Load the weights
        network_path = f"models/pretrained_models/{network_name}.th"

        load_from_dict(network=network, device=device, path=network_path)
    elif network_name == "EfficientNet":
        network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif "VGG" in network_name:
        if network_name == "VGG11":
            network = vgg11(pretrained=True)
        network_path = f'models/pretrained_models/cifar10/{network_name}.th'

        load_from_dict(network=network,
                       device=device,
                       path=network_path)
        
    elif 'DenseNet' in network_name:
        if network_name == 'DenseNet121':
            network = densenet121()
        elif network_name == 'DenseNet161':
            network = densenet161()
        elif network_name == 'DenseNet169':
            network = densenet169()
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

         
        network_path = f'models/pretrained_models/cifar10/{network_name}.pt'
        
        load_from_dict(network=network,
                       device=device,
                       path=network_path)
    
    elif 'Vgg' in network_name:
        if network_name == 'Vgg11_bn':
            network = vgg11_bn()
        elif network_name == 'Vgg13_bn':
            network = vgg13_bn()
        elif network_name == 'Vgg16_bn':
            network = vgg16_bn()
        elif network_name == 'Vgg19_bn':
            network = vgg19_bn()
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

         
        network_path = f'models/pretrained_models/cifar10/{network_name}.pt'
        
        load_from_dict(network=network,
                       device=device,
                       path=network_path)
        
    
    elif 'GoogLeNet' in network_name:
        network = GoogLeNet()
        network_path = f'models/pretrained_models/cifar10/{network_name}.pt'
        
        load_from_dict(network=network,
                       device=device,
                       path=network_path)
        
    elif 'MobileNetV2' in network_name:
        network = MobileNetV2()
        network_path = f'models/pretrained_models/cifar10/{network_name}.pt'
        
        state_dict = torch.load(network_path, map_location=device)['net']
        function = None
        if function is None:
            clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        else:
            clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

        network.load_state_dict(clean_state_dict, strict=False)
        
    elif 'InceptionV3' in network_name:
        network = Inception3()
        network_path = f'models/pretrained_models/cifar10/{network_name}.pt'
        
        load_from_dict(network=network,
                       device=device,
                       path=network_path)
        
        
    else:
        raise UnknownNetworkException(f"ERROR: unknown network: {network_name}")

    # Send network to device and set for inference
    network.to(device)
    network.eval()

    return network


def get_device(forbid_cuda: bool, use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
        if use_cuda:
            print("WARNING: cuda forcibly disabled even if set_cuda is set")
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = ""
                print("ERROR: cuda not available even if use-cuda is set")
                exit(-1)
        else:
            device = "cpu"

    return torch.device(device)

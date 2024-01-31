import os
import argparse

import torch
from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets, Load_CIFAR100_datasets, Load_GTSRB_datasets
from models.utils import load_from_dict

from models.CIFAR10 import inception_cifar10
from models.CIFAR10 import mobilenetv2_cifar10
from models.CIFAR10 import googlenet_cifar10
from models.CIFAR10 import mobilenetv2_cifar10
from models.CIFAR10 import vgg_cifar10
from models.CIFAR10 import resnet_cifar10
from models.CIFAR10 import densenet_cifar10

from models.CIFAR100 import resnet_cifar100
from models.CIFAR100 import densenet_cifar100
from models.CIFAR100 import googlenet_cifar100

from models.GTSRB import vgg_GTSRB
from models.GTSRB import resnet_GTSRB
from models.GTSRB import densenet_GTSRB

import models.imagenet.Vgg_imagenet as vgg_imagenet



# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader


class UnknownNetworkException(Exception):
    pass


SUPPORTED_MODELS_LIST = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
    "ResNet1202",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "GoogLeNet",
    "MobileNetV2",
    "InceptionV3",
    "Vgg11_bn",
    "Vgg13_bn",
    "Vgg16_bn",
    "Vgg19_bn",
#    "Vgg11_ImageNet"
]


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(
        description="Run Inferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', '-d', type=str, 
                        help='Dataset to use',
                        choices=['CIFAR10','CIFAR100','GTSRB', 'IMAGENET'],)
    parser.add_argument(
        "--forbid-cuda",
        action="store_true",
        help="Completely disable the usage of CUDA. This command overrides any other gpu options.",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use the gpu if available."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Test set batch size"
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS_LIST,
    )
    parser.add_argument(
        "--tensorflow",
        "--tf",
        help="Execute the network in TensorFlow. Convert it a conversion does not exists",
        action="store_true",
    )
    parsed_args = parser.parse_args()

    return parsed_args


def get_loader(
    network_name: str,
    dataset_name:str,
    batch_size: int,
    image_per_class: int = None,
    network: torch.nn.Module = None,
    permute_tf=False,
) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if dataset_name == 'CIFAR10':
        print("Loading CIFAR10 dataset")
        train_loader, _, loader = load_CIFAR10_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
            permute_tf=permute_tf,
        )

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
    elif dataset_name == 'IMAGENET':
        if image_per_class is None:
            image_per_class = 5
        loader = load_ImageNet_validation_set(
            batch_size=batch_size,
            image_per_class=image_per_class,
            network=network,
        )
        train_loader = load_ImageNet_validation_set(
            batch_size=batch_size,
            image_per_class=None,
            network=network,
        )
    
    elif dataset_name == 'CIFAR100':
        print("Loading CIFAR100 dataset")
        train_loader, _, loader = Load_CIFAR100_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
        )
    
    elif dataset_name == 'GTSRB':
        print("Loading GTSRB dataset")
        train_loader, _, loader = Load_GTSRB_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
        )
    
    else:
        raise UnknownNetworkException(f"ERROR: unknown dataset: {dataset_name}")

    print(f"Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}")

    return train_loader, loader


def load_network(network_name: str, device: torch.device, dataset_name: str) -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :return: The loaded network
    """
    if dataset_name == 'CIFAR10':
        if "ResNet" in network_name:
            if network_name == "ResNet20":
                network_function = resnet_cifar10.resnet20()
            elif network_name == "ResNet32":
                network_function = resnet_cifar10.resnet32()
            elif network_name == "ResNet44":
                network_function = resnet_cifar10.resnet44()
            elif network_name == "ResNet56":
                network_function = resnet_cifar10.resnet56()
            elif network_name == "ResNet110":
                network_function = resnet_cifar10.resnet110()
            elif network_name == "ResNet1202":
                network_function = resnet_cifar10.resnet1202()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )

            network = network_function
            

            # Load the weights
            network_path = f"models/pretrained_models/CIFAR10/{network_name}.th"

            load_from_dict(network=network, device=device, path=network_path)
            
        
        elif "Vgg" and "ImageNet" in network_name:
            if network_name == "Vgg11_ImageNet":
                network = vgg_imagenet.vgg11(pretrained=True)
                
        elif "DenseNet" in network_name:
            if network_name == "DenseNet121":
                network = densenet_cifar10.densenet121()
            elif network_name == "DenseNet161":
                network = densenet_cifar10.densenet161()
            elif network_name == "DenseNet169":
                network = densenet_cifar10.densenet169()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )

            network_path = f"models/pretrained_models/CIFAR10/{network_name}.pt"

            load_from_dict(network=network, device=device, path=network_path)

        elif "Vgg" in network_name:
            if network_name == "Vgg11_bn":
                network = vgg_cifar10.vgg11_bn()
            elif network_name == "Vgg13_bn":
                network = vgg_cifar10.vgg13_bn()
            elif network_name == "Vgg16_bn":
                network = vgg_cifar10.vgg16_bn()
            elif network_name == "Vgg19_bn":
                network = vgg_cifar10.vgg19_bn()
            else:
                raise UnknownNetworkException(
                    f"ERROR: unknown version of ResNet: {network_name}"
                )

            network_path = f"models/pretrained_models/CIFAR10/{network_name}.pt"

            load_from_dict(network=network, device=device, path=network_path)

        elif "GoogLeNet" in network_name:
            network = googlenet_cifar10.GoogLeNet()
            network_path = f"models/pretrained_models/CIFAR10/{network_name}.pt"

            load_from_dict(network=network, device=device, path=network_path)

        elif "MobileNetV2" in network_name:
            network = mobilenetv2_cifar10.MobileNetV2()
            network_path = f"models/pretrained_models/CIFAR10/{network_name}.pt"

            state_dict = torch.load(network_path, map_location=device)["net"]
            function = None
            if function is None:
                clean_state_dict = {
                    key.replace("module.", ""): value for key, value in state_dict.items()
                }
            else:
                clean_state_dict = {
                    key.replace("module.", ""): function(value)
                    if not (("bn" in key) and ("weight" in key))
                    else value
                    for key, value in state_dict.items()
                }

            network.load_state_dict(clean_state_dict, strict=False)

        elif "InceptionV3" in network_name:
            network = inception_cifar10.Inception3()
            network_path = f"models/pretrained_models/CIFAR10/{network_name}.pt"

            load_from_dict(network=network, device=device, path=network_path)

        else:
            raise UnknownNetworkException(f"ERROR: unknown network: {network_name}")
    
    
    elif dataset_name == 'CIFAR100':
        
        
        print(f'Loading network {network_name}')    
        if 'ResNet' in network_name:  
            network = resnet_cifar100.resnet18()
            print('resnet18 loaded')
        elif 'DenseNet' in network_name:  
            network = densenet_cifar100.densenet121()
            print('densenet121 loaded')
        elif 'GoogLeNet' in network_name:
            network = googlenet_cifar100.googlenet()
            print('googlenet loaded')
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of the model: {network_name}')
        
        network_path = f'models/pretrained_models/{dataset_name}/{network_name}_{dataset_name}.pth'

        load_from_dict(network=network,
                        device=device,
                        path=network_path)
        
        
    elif dataset_name == 'GTSRB':
        
        print(f'Loading network {network_name}')
        if 'ResNet' in network_name:
            network = resnet_GTSRB.resnet20()
            print('resnet20 loaded')
        elif 'DenseNet' in network_name:
            network = densenet_GTSRB.densenet121()
            print('densenet121 loaded')
        elif 'Vgg' in network_name:
            network = vgg_GTSRB.vgg11_bn()
            print('vgg11_bn loaded')
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of the model: {network_name}')
        
        network_path = f'models/pretrained_models/{dataset_name}/{network_name}_{dataset_name}.pt'
        load_from_dict(network=network,
                        device=device,
                        path=network_path)
        
  
    else:
        raise UnknownNetworkException(f'ERROR: unknown dataset: {dataset_name}')
        
    network.to(device)
    network.eval()

    # Send network to device and set for inference
    

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

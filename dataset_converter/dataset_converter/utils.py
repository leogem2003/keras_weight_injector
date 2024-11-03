from dataset_converter.dataset_loader import (
    load_CIFAR10_datasets,
    load_CIFAR100_datasets,
    load_GTSRB_datasets,
)
import importlib.resources

# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader


class UnknownNetworkException(Exception):
    pass


MODULE_PATH = importlib.resources.files(__package__)

SUPPORTED_DATASETS = ["CIFAR10", "CIFAR100", "GTSRB"]

SUPPORTED_MODELS = [
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
]


def get_loader(
    dataset_name: str,
    batch_size: int,
    image_per_class: int = None,
    permute_tf=False,
) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param dataset_name: The name of the dataset
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :return: The DataLoader
    """
    if dataset_name == "CIFAR10":
        print("Loading CIFAR10 dataset")
        train_loader, _, loader = load_CIFAR10_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
            permute_tf=permute_tf,
        )
    elif dataset_name == "CIFAR100":
        print("Loading CIFAR100 dataset")
        train_loader, _, loader = load_CIFAR100_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
            permute_tf=permute_tf,
        )

    elif dataset_name == "GTSRB":
        print("Loading GTSRB dataset")
        train_loader, _, loader = load_GTSRB_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class,
            permute_tf=permute_tf,
        )

    else:
        raise UnknownNetworkException(f"ERROR: unknown dataset: {dataset_name}")

    print(f"Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}")

    return train_loader, loader

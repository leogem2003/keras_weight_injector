import torch

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets

from utils import load_network, get_device, parse_args
from classes_core.error_simulator_keras import create_injection_sites_layer_simulator, ErrorSimulator


def main(args):
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=False)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda, use_cuda=args.use_cuda)

    print(f"Using device {device}")

    # Load the dataset
    if "ResNet" in args.network_name:
        _, _, loader = load_CIFAR10_datasets(
            test_batch_size=args.batch_size, permute_tf=args.tensorflow
        )
        print(f"Using dataset: CIFAR10")

    else:
        loader = load_ImageNet_validation_set(
            batch_size=args.batch_size, image_per_class=1, permute_tf=args.tensorflow
        )
    if args.tensorflow:
        # Import inference manager only here to avoid importing tensorflow for pytorch users
        from TFInferenceManager import TFInferenceManager
        from tf_utils import load_converted_tf_network, clone_model
        import keras

        tf_network = load_converted_tf_network(args.network_name)

        tf_network.summary(expand_nested=True)

        temp_tf_network = keras.models.clone_model(tf_network)
        temp_tf_network.set_weights(tf_network.get_weights())

        def factory(layer, inputs):
            if False:  # layer.name == 'conv2d':
                return keras.layers.ReLU()
            else:
                return None
            
        def classes_factory(layer, inputs, output):
            if layer.name == 'conv2d_3':
                print(output.shape)
                n, h, w, c = output.shape
                available_injection_sites, masks = create_injection_sites_layer_simulator(
                    5,
                    'conv_gemm',
                    str((1, c, h, w)),
                    str((1, h, w, c)),
                    models_folder='classes_models'
                )

                sim = ErrorSimulator(available_injection_sites, masks, len(available_injection_sites), [0])
                return sim
            else:
                return None

        cloned_model = clone_model(
            temp_tf_network,
            temp_tf_network.layers[0],
            layer_factory=classes_factory,
            verbose=True,
            copy_weights=True
        )

        cloned_model.summary()

        # Execute the fault injection campaign with the smart network
        inference_executor = TFInferenceManager(
            network=cloned_model, network_name=args.network_name, loader=loader
        )

    else:
        # Import inference manager only here to avoid importing pytorch for tensorflow users

        from InferenceManager import InferenceManager

        # Load the network
        network = load_network(network_name=args.network_name, device=device)

        print(f"Using network: {args.network_name}")

        network.eval()
        # Execute the fault injection campaign with the smart network
        inference_executor = InferenceManager(
            network=network,
            device=device,
            network_name=args.network_name,
            loader=loader,
        )

    # This function runs clean inferences on the golden dataset
    inference_executor.run_clean()


if __name__ == "__main__":
    main(args=parse_args())

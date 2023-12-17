import torch

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets

from utils import load_network, get_device, parse_args

def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=False)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    
    print(f'Using device {device}')

    # Load the dataset
    if 'ResNet' in args.network_name:
        _, _, loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)
        print(f"Using dataset: CIFAR10")
        
    else:
        loader = load_ImageNet_validation_set(batch_size=args.batch_size,
                                              image_per_class=1)
    if args.tensorflow:
        # Import inference manager only here to avoid importing tensorflow for pytorch users
        from TFInferenceManager import TFInferenceManager
        from tf_utils import load_converted_tf_network
        tf_network = load_converted_tf_network(args.network_name)

        # Execute the fault injection campaign with the smart network
        inference_executor = TFInferenceManager(network=tf_network,
                                                network_name=args.network_name,
                                                loader=loader)
    else:
        # Import inference manager only here to avoid importing pytorch for tensorflow users

        from InferenceManager import InferenceManager

        # Load the network
        network = load_network(network_name=args.network_name,
                            device=device)
        
        print(f"Using network: {args.network_name}")
        
        network.eval()
        # Execute the fault injection campaign with the smart network
        inference_executor = InferenceManager(network=network,
                                              device=device,
                                              network_name=args.network_name,
                                              loader=loader)
    
    #This function runs clean inferences on the golden dataset
    inference_executor.run_clean()

if __name__ == '__main__':
    main(args=parse_args())

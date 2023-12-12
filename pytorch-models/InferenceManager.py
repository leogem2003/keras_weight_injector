import os
import shutil
import time
import math
from datetime import timedelta

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class InferenceManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader):

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        # The clean output of the network after the first run
        self.clean_output_scores = list()
        self.clean_output_indices = list()
        self.clean_labels = list()

        # The output dir
        self.label_output_dir = f'output/{self.network_name}/pt/label/batch_size_{self.loader.batch_size}'
        self.clean_output_dir = f'output/{self.network_name}/pt/clean/batch_size_{self.loader.batch_size}'

        # Create the output dir
        os.makedirs(self.label_output_dir, exist_ok=True)
        os.makedirs(self.clean_output_dir, exist_ok=True)


        
    def run_clean(self):
        """
        Run a clean inference of the network
        :return: A string containing the formatted time elapsed from the beginning to the end of the fault injection
        campaign
        """

        with torch.no_grad():

            # Start measuring the time elapsed
            start_time = time.time()

            # Cycle all the batches in the data loader
            pbar = tqdm(self.loader,
                        colour='green',
                        desc=f'Clean Run',
                        ncols=shutil.get_terminal_size().columns)
            dataset_size=0

            for batch_id, batch in enumerate(pbar):
                #print(batch_id)
                data, label = batch
                #print(len(label)) #total of 10000 images
                #print(label)
                dataset_size=dataset_size+len(label)
                data = data.to(self.device)

                # Run inference on the current batch
                scores, indices = self.__run_inference_on_batch(data=data)


                # Save the output
                torch.save(scores, f'{self.clean_output_dir}/batch_{batch_id}.pt')
                torch.save(label, f'{self.label_output_dir}/batch_{batch_id}.pt')

                # Append the results to a list
                self.clean_output_scores.append(scores)
                self.clean_output_indices.append(indices)
                self.clean_labels.append(label)

        # COMPUTE THE ACCURACY OF THE NEURAL NETWORK       
        # Element-wise comparison for each pair of lists
  
        elementwise_comparison = [label != index for labels, indices in zip(self.clean_labels, self.clean_output_indices) for label, index in zip(labels, indices)]          
        
        # Count the number of True values in the list
        num_different_elements = elementwise_comparison.count(True)
        print(f"The DNN wrong predicions are: {num_different_elements}")
        accuracy= (1 - num_different_elements/dataset_size)*100
        print(f"The final accuracy is: {accuracy}%")
        
        # Stop measuring the time
        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed))


    def __run_inference_on_batch(self,
                                 data: torch.Tensor):
        """
        Rim a fault injection on a single batch
        :param data: The input data from the batch
        :return: a tuple (scores, indices) where the scores are the vector score of each element in the batch and the
        indices are the argmax of the vector score
        """

        # Execute the network on the batch
        network_output = self.network(data) # it is a vector of output elements (one vector for each image). The size is num_batches * num_outputs
        #print(network_output)  
        prediction = torch.topk(network_output, k=1)  # it returns two lists : values with the top1 values and indices with the indices
        #print(prediction.indices)
        
 
        # Get the score and the indices of the predictions
        prediction_scores = network_output.cpu()
        
        prediction_indices = [int(fault) for fault in prediction.indices]
        return prediction_scores, prediction_indices


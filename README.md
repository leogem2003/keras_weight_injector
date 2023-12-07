# How to run DNN inferences

A clean Pytorch inference can be executed with the following programm:
```
python main.py -n network-name -b batch-size 
```

Available models are : 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202'

It is possible to execute inferences with available GPUs sepcifing the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/network_name/pt``` folder. 

*Note*: The progress bar shows the percentage of predictions that have chagned as a result of a fault. THIS IS NOT A MEASURE OF ACCURACY LOSS, even if it is related. The beavhoiur cna be changed to check differences in vector score rather than in predicitons.

# Requirements 

Before running inferences, install the needed packages and tools
```
pip install -r requirements.txt
```


# Analyse the results: from .pt to .csv

Results file can be converted to csv using the script:
```
python pt_to_csv.py -n network-name -b batch-size 
```
Results are saved in the ```output/network_name/csv``` folder. Notice that carrying out operation on the CSV file is going to be more expensive than carrying out the same analysis on .pt files. This format should be used only for data visualization purposes only.


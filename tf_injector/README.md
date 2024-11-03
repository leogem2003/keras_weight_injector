# tf_injector
Fault injector for Tensorflow2 modules.

## Setup
1. Ensure you have Python 3.10 installed in your working environment
2. Create a virtual environment
```
python -m venv venv_name
source venv_name/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
## Usage
Run as a python module:
```
python -m tf_injector [ARGS, ...]
```
To display the complete usage guide, type
```
python -m tf_injector --help
```
## Input
To run an injection campaign, you will need:
- A target network saved in `.keras` format, saved in `models/<dataset>/<network>.keras`
- A dataset compatible with tensorflow datasets, saved in `datasets/<dataset>` 
- A fault list compatible with the target network in csv format (header needed)

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |


## Outputs
By default, a summarized report of the injection campaign is stored in reports/<dataset>/<network>/<dataset>_<network>_<datetime>.csv
By enabling the --save-outputs flag, inference outputs are saved as numpy arrays in the same folder, as <datetime>/(clean | (fault_<inj_id)).npy

### Output metrics

- part 1: Injection info

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |

- part 2: robustness
    - `top_1_correct`: the label with the maximum score equals the test label (correct inference)
    - `top_5_correct`: the test label is inside the set of the labels which gained the five highest scores
    - `top_1_robust`: Same as top_1_correct, but compared with the predicted labels of the golden inference 
    - `top_5_robust`: Same as top__correct, but compared with the predicted labels of the golden inference 


| top_1_correct | top_5_correct | top_1_robust | top_5_robust |
| --------------- | --------------- | --------------- | ------------ |
| 9174 | 9977 | 10000 | 10000 |
| 9174 | 9977 | 10000 | 10000 |

- part 3: other stats
    - `masked`: Number of dataset inferences that identified the fault as masked.
    - `non_critical`: Number of dataset inferences that identified the fault as non-critical.
    - `critical`: Number of dataset inferences that identified the fault as critical (SDC-1).

| n_injections | masked | non_critical | critical |
|:------------:|:------:|:------------:|:--------:|
|        10000 |  10000 |            0 |        0 |
|        10000 |  10000 |            3 |        0 |


# Acknowledgments

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them.

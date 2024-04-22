# Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks

This is the official repository for the "Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks" paper, currently under review at IEEE TIFS (momentarely available on arXiv: [here](https://arxiv.org/abs/2403.11830)).

If you use any part of this codebase, you are kindly invited to cite our paper:
```
@article{venturi2024problem,
  title={Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks},
  author={Venturi, Andrea and Stabili, Dario and Marchetti, Mirco},
  journal={arXiv preprint arXiv:2403.11830},
  year={2024}
}
```

## Organization
This README is structured into the following sections:
- [Disclaimer](#disclaimer)
- [Contents](#contents)
- [Instructions](#instructions)
- [Thanks](#thanks)
  

## Disclaimer
Due to space and copyright constraints, this repository does not include all datasets used in our study, which are publicly available here: [CTU-13](https://www.stratosphereips.org/datasets-ctu13) and [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets).

To facilitate replication of our experiments, we provide the full codebase used in the research and a sample of the CTU-13 dataset. Specifically, we include malicious netflows for the `menti` botnet and benign netflows necessary for training and testing the GNN-based detection models. We extend our gratitude to the dataset authors for allowing us to share this sample.

It is likely that a "novel" experiment leads to results that are slighlty different from the ones reported in the paper. This is normal, as preprocessing operations might be different from time to time. Moreover, in the paper we report the average results over a sufficiently large number of executions.

Feel free to contact me for any issue (andrea.venturi@unimore.it).

## Contents

The repository is organized as follows: 
```
.
├── datasets                    # Datasets directory
│   ├── CTU
│   └── ToN-IoT
├── figures                     # Figures obtained from the results
│   ├── CTU
│   ├── ToN_IoT
│   └── total
├── preprocessed_data           # Datasets after preprocessing
│   ├── CTU
│   └── ToN_IoT
└── results                     # Results directory
    ├── CTU
    └── ToN_IoT
```
We first explain the content of the directories and then we present each src file.

### Directories

#### datasets folder
This directory contains all the raw datasets referenced in our paper and is organized into two subdirectories: CTU and ToN-IoT. Each directory contains collections of raw netflows.

**IMPORTANT:** We have organized the netflows by attack type within each dataset, resulting in separate CSV files for each attack, along with a single CSV file for all benign netflows. This organization simplifies data handling. To assist with separating the raw data accordingly, we provide a script named `separate_data.py`.

The naming convention for each file is `<attack_name>.csv`, where `<attack_name>` represents the specific attack type of the netflows contained within the file. In the CTU dataset, the attack names include `neris, rbot, virut, menti, murlo`. In the ToN-IoT dataset, they include `backdoor, ddos, dos, injection, password, ransomware, scanning, xss`.

#### figures folder
This folder contains the figures that illustrate the results as reported in the paper.

#### preprocessed_data folder
This folder contains the datasets after the execution of the `preprocessing.py` and `preprocessing_ToN.py` scripts.


#### results folder
This folder contains the results for our experiments.
The organization of this directory is as follows:
```
.
├── CTU                                                       # results for CTU
│   ├── models                                                # trained models
│   │   ├── egraphsage                                        # E-GraphSAGE
│   │   │   ├── <attack_name>                                 # Specific model instance
│   │   │   │   ├── dict.pth                  
│   │   │   │   ├── model.pth
│   │   │   │   └── scaler.skl
│   │   │   └── ...
│   │   └── linegraphsage                                     # LineGraphSAGE
│   │       ├── <attack_name>                                 # LineGraphSAGE instance
│   │       │   ├── dict.pth
│   │       │   ├── model.pth
│   │       │   └── scaler.skl
│   │       └── ...
│   └── scores                                                # score results
│       ├── egraphsage                                        # Score for E-GraphSAGE
│       │   ├── <attack_name>                                 # Specific model instance
│       │   │   ├── aa_feature_f1.csv                         # Results for feature atks
│       │   │   ├── aa_feature_precision.csv                  
│       │   │   ├── aa_feature_recall.csv
│       │   │   ├── aa_structure_add_node.csv                 # Add Node attack (DR)
│       │   │   ├── aa_structure_benign_from_C.csv            # C2X_B attack (DR)
│       │   │   ├── aa_structure_malicious_from_C.csv         # C2X_M attack (DR)
│       │   │   └── baseline.csv                              # Baseline results
│       │   └── ...
│       ├── linegraphsage                                     # LineGraphSAGE
│       │   ├── <attack_name>                                 # LineGraphSAGE instance
│       │   │   ├── aa_feature_f1.csv
│       │   │   ├── aa_feature_precision.csv
│       │   │   ├── aa_feature_recall.csv
│       │   │   ├── aa_structure_add_node.csv
│       │   │   ├── aa_structure_benign_from_C.csv
│       │   │   ├── aa_structure_malicious_from_C.csv
│       │   │   └── baseline.csv
│       │   └── ...
│       └── rf                                                # Random Forest
│           ├── <attack_name>                                 # Specific model
│           │   ├── baseline.csv                              # Baseline results
│           │   └── feature_attack.csv                        # Results for feature atks
│           └── ...
└── ToN_IoT                               # Same structure as above
    ├── models
    |   └── ... 
    └── scores
        └── ...
```
This structure is automatically created at the execution of the scripts.

### Scripts
- `preprocessing[_ToN].py`: This file contains the code for preprocessing the raw dataset files
- `EGraphSAGE.py`: This file contains the implementation of the E-GraphSAGE model. The code is from the official [E-GraphSAGE repository](https://github.com/waimorris/E-GraphSAGE)
- `<train|test>_egraphsage.py`: This file contains the code for training and testing the E-GraphSAGE model
- `batch_<train|test>_linegraphsage.py`: This file contains the code for training and testing the LineGraphSAGE model. 
- `adversarial_feature.py`: This file contains the implementation of feature-based attacks.
- `adversarial_structure.py`: This file contains the implementation of structural attacks.
- `baseline_test_<model>.py`: This file contains the code for testing the GNN models in baseline, feature attack and structural attack scenarios.
- `plot.ipynb`: This notebook contains the procedures to reproduce the figures of the paper
- `randomforest.ipynb`: This notebook contains the code for training and testing the RandomForest classifier
- `utils.py`: This file contains some utility functions used throughout the entire codebase.

## Instructions
To easily reproduce all the experiments, run the `all_experiments.sh` script. If you only wish to perform specific experiments, modify the `all_experiments.sh` file by commenting out the lines for experiments you do not need.

Alternatively, you can follow this straightforward workflow:

1. **Get the data**: Download the datasets from [CTU-13](https://www.stratosphereips.org/datasets-ctu13) and [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets). Store the files in the appropriate dataset directories. **IMPORTANT**: Organize netflow collections by attack as specified in [Contents](#contents). Alternatively, use the provided CTU13 dataset snippet from [datasets/CTU/data.zip](datasets/CTU/data.zip). Unzip and place the contents in the `datasets/CTU` directory to set up the data for training GNN models to detect `menti` botnet traffic.
2. **Preprocess the data**: Execute `preprocessing.py` for the CTU dataset and `preprocessing_ToN.py` for the ToN-IoT dataset.
3. **Train the model**: Choose either the E-GraphSAGE model (`train_egraphsage.py`) or the LineGraphSAGE model (`batch_train_linegraphsage.py`), and execute it.
4. **Perform baseline tests**: Test the models in non-adversarial settings executing `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` with the `--test` option to specify the test dataset.
5. **Execute feature attacks**: Conduct feature-based attacks by running the `baseline_test_<egraphsage|linegraphsage>.py` scripts with the `--feature_attack` option.
6. **Perform $C2X_B$ attack**. Launch the $C2X_B$ structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack benign_from_C`.
7. **Perform $C2X_M$ attack**. Launch the $C2X_M$ structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack malicious_from_C`.
8. **Perform the add_node attack**. Launch the add_node structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack add_node`.

## Thanks

Kudos to the authors of:

#### E-GraphSAGE
[Paper] (https://ieeexplore.ieee.org/abstract/document/9789878) [Repo](https://github.com/waimorris/E-GraphSAGE)

Reference:
```
@inproceedings{lo2022graphsage,
  title={E-graphsage: A graph neural network based intrusion detection system for iot},
  author={Lo, Wai Weng and Layeghy, Siamak and Sarhan, Mohanad and Gallagher, Marcus and Portmann, Marius},
  booktitle={NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium},
  pages={1--9},
  year={2022},
  organization={IEEE}
}
```

#### CTU-13 dataset
Homepage: https://www.stratosphereips.org/datasets-

Reference:
```
@article{garcia2014empirical,
  title={An empirical comparison of botnet detection methods},
  author={Garcia, Sebastian and Grill, Martin and Stiborek, Jan and Zunino, Alejandro},
  journal={computers \& security},
  volume={45},
  pages={100--123},
  year={2014},
  publisher={Elsevier}
}
```

#### ToN-IoT dataset
Homepage: https://research.unsw.edu.au/projects/toniot-datasets

```
@article{alsaedi2020ton_iot,
  title={TON\_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems},
  author={Alsaedi, Abdullah and Moustafa, Nour and Tari, Zahir and Mahmood, Abdun and Anwar, Adnan},
  journal={Ieee Access},
  volume={8},
  pages={165130--165150},
  year={2020},
  publisher={IEEE}
}
```
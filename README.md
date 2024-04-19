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

## Disclaimer
For reasons of space and copyright we cannot include in this repo all the considered datasets used in the paper. Nevertheless, we provide the entire codebase to reproduce the results starting from the data as provided by the datasets [CTU-13](https://www.stratosphereips.org/datasets-ctu13) and [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets). Feel free to contact me for any issue (andrea.venturi@unimore.it).

It is likely that a "novel" experiment leads to results that are slighlty different from the ones reported in the paper. This is normal, as preprocessing operations might be different from time to time. Moreover, in the paper we report the average results over a sufficiently large number of executions.

## Organization
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
    └──  ToN_IoT
```
## Contents




## Instructions
The simplest way to reproduce all the experiments is by executing the `all_experiments.sh` script. You can also select a single experiment by commenting the unnecessary lines in the `all_experiments.sh` file.

Otherwise, you can follow this simple workflow:

1. **Get the data**. Download the data from the datasets homepages [CTU-13](https://www.stratosphereips.org/datasets-ctu13) and [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets) and place the files in the corresponding dataset directory. 
2. **Preprocess data**. Use the `preprocessing.py` and `preprocessing_ToN` files respectively for the CTU dataset and the ToN-IoT dataset.
3. **Train the model**. Again you can choose between the E-GraphSAGE model (`train_egraphsage.py` file) and linegraphsage (`batch_train_linegraphsage.py`).
4. **Perform the baseline tests**. Test the models in non-adversarial scenarios (no perturbations). Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`).
5. **Perform the feature attacks**. Launch the feature-based attacks against the models. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--feature_attack`.
6. **Perform $C2X_B$ attack**. Launch the $C2X_B$ structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack benign_from_C`.
7. **Perform $C2X_M$ attack**. Launch the $C2X_M$ structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack malicious_from_C`.
8. **Perform the add_node attack**. Launch the add_node structural attack. Execute the `baseline_test_egraphsage.py` or `baseline_test_linegraph.py` files with argument `--test` indicating the desired test dataset (e.g., `<preproc_directory>/<dataset>/<attack>_test.csv`), and option `--structure_attack add_node`.

## Thanks

Kudos to the authors of the E-GraphSAGE repo: [Check it out!](https://github.com/waimorris/E-GraphSAGE)


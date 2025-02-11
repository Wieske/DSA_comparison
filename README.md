# Dynamic Survival Analysis

This repository can be used to reproduce the results from "A Comparative Study of Methods for Dynamic Survival Analysis" by Wieske de Swart, Marco Loog and Jesse Krijthe.

The repository contains the following files and directories:
```
├── dataset: contains the synthetic data
├── figures: contains the figures shown in the paper
├── results: contains combined results from all experiments
├── skeleton: contains python scripts
├── data_processing.py: used for preprocessing of the ADNI TADPOLE dataset
├── environment.yml: can be used to reproduce the environment in conda
├── main.py: main python script
├── main_random_search.py: main python script for performing random hyperparameter search
├── paper_figures.py: can be used to reproduce all figures from the paper and supplementary material
├── requirements.txt: can be used to reproduce the environment
```

All figures in the paper can be directly reproduced with the script `paper_figures.py`, 
which will load the results from the results directory.
For this part only the python packages `pandas` and `matplotlib` are required. 

The following sections will describe how to reproduce the simulation and ADNI experiments. 
For this a python environment should be created using either `environment.yml` (for conda) or `requirements.txt`. 
This will also install an adapted version of the FDApy package for the MFPCA model, which can be found [here](https://github.com/Wieske/FDApy/tree/mfpca-pace-update).

## Simulation experiments
The simulation experiments can be reproduced using the data contained in the dataset directory of this repository. 
This data was created using the function `save_simdatasets(trainseed=42, testseed=0)` from the script `data_simulation.py`.

The simulation experiments can be reproduced by running the main script with the following parameters (for scenario 3):
```bash
main.py 
--project "Landmarking_s3_1000" \
--filepath "dataset/train/simdata_s3.csv" \
--test_file "dataset/test/simdata_s3.csv" \
--train_size 1000 \
--train_sets 10 \
--long_model $longitudinal_model \
--train_landmarking $landmarking_method
```
with `$longitudinal_model` one of `"baseline"`, `"last_visit"`, `"MFPCA"`, `"RNN"` or `"RNN_long"`
and `$landmarking_method` one of `"None"`, `"super"`, `"random"` or `"strict"`

## ADNI experiments
For a real-world application data is used from the Alzheimer's Disease Neuroimaging Initiative (ADNI). 
Access to this data can be requested at https://adni.loni.usc.edu/data-samples/adni-data/#AccessData.
Once access is granted the results can be reproduced by following these steps:
* Download the Tadpole Challenge Data (in Study Files: Test Data)
* Copy the file "TADPOLE_D1_D2.csv" to the dataset directory
* Run the script `data_preprocessing.py` to prepare the data
* Run the main script with the following parameters
```bash
main.py 
--project "Landmarking_ADNI" \
--filepath "dataset/df_adni_tadpole.csv" \
--cross_validation True \
--train_sets 10 \
--missing_impute "ffill" \
--normalization "standard" \
--long_model $longitudinal_model \
--train_landmarking $landmarking_method 
```
with `$longitudinal_model` one of `"baseline"`, `"last_visit"`, `"MFPCA"`, `"RNN"` or `"RNN_long"`
and `$landmarking_method` one of `"None"`, `"super"`, `"random"` or `"strict"`

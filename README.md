# Fair Multilingual Vandalism Detection System for Wikipedia

<img align="right" src="https://upload.wikimedia.org/wikipedia/commons/1/17/System_design_inference.png" alt="drawing" style="width:300px;"/>

This repository includes resources to reproduce training procedures for the paper
**Fair Multilingual Vandalism Detection System for Wikipedia** from data collection to model training. 

- The full paper already available: [![DOI:10.1145/3580305.3599823](https://zenodo.org/badge/DOI/10.1145/3580305.3599823.svg)](
https://doi.org/10.1145/3580305.3599823)
- The model inference logic is implemented in 
[![Knowledge Integrity repo](https://img.shields.io/badge/GitLab-repo-orange)](https://gitlab.wikimedia.org/repos/research/knowledge_integrity)
- Prepared dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8174336.svg)](https://doi.org/10.5281/zenodo.8174336)

## Experiments reproducing:

### Simplified experiment:
We publish the [dataset](https://doi.org/10.5281/zenodo.8174336) that can be directly used for final classification model training. 
Data includes also raw features, that are not directly used in the final classification model like texts.

The training script that uses mentioned collected datasets:
```commandline
python modules/train_model_simplified.py --train=data/train_anon_users.csv --test=data/test_anon_user
s.csv --name=anon_model
```

### Full experiment: 
#### 1. Data collection:
Data collection logic was done using Wikimedia analytical cluster. 
How to collect data on the cluster:
```commandline
for i in ka lv ta ur eo lt sl hy hr sk eu et ms az da bg sr ro el th bn simple no hi ca hu ko fi vi uz sv cs he id tr uk nl pl ar fa it zh ru es ja de fr en
do
   python modules/data_collection.py -l $i -m train -f1 0 -f2 0 -n 300000 -uc 1 -p all_users
done
```
The script above collects the dataset for each of the given languages in the loop. 
It does not filter anonymous users (-f1) and revision wars (-f2). 
Also, there is a limit of 300000 records per language.

Data collection of hold-out test:
```commandline
for i in ka lv ta ur eo lt sl hy hr sk eu et ms az da bg sr ro el th bn simple no hi ca hu ko fi vi uz sv cs he id tr uk nl pl ar fa it zh ru es ja de fr en
do
   python modules/data_collection.py -l $i -m test -f1 0 -f2 1 -n 100000 -uc 1 -p full_test
done
```
The sample for collected data can be found here: .....link......
Full data will be released soon.

#### 2. Initial processing:
Having the data for each language, the next step is preparing it for training.
For that purposes, run:
```commandline
python modules/data_prepareration.py
```
This script load data for each language in the defined list, 
do the timestamp-based train-test split, 
splitting train for MLMs fine-tuning and classifier model,
setting the balancing key for further evaluation, 
filtering revision wars if it is needed.

It returns three aggregated dataframes: train, test, full-test (independent set)

#### 3. MLM tuning:
Having collected the training dataset, we proceed with fine-tuning of MLMs. 
```commandline
python modules/feature_trainer.py
```

This script prepares the specific dataset for four different MLMs model tuning.
Later it uses prepared data and tunes MLM for a pair of text classification (changes),
text classification (inserts, removes), and regression (title semantics). And later tune those 
models and save them to the \models directory.
GPU is needed for training. We used AMD Radeon Pro WX 9100 16GB GPU.


#### 4. MLM features calculation for datasets
Having collected the training dataset and tuned MLMs we proceed with calculation text features for 
those datasets (training and testing parts)
```commandline
python modules/feature_builder.py
```
Before running the script, we should make sure that paths to MLMs and datasets are correct in the script.

#### 5. User features extraction for datasets
Additional user features extraction (is_anonymous, user group):
```commandline
python modules/users_features_collection.py
```
The module takes as input the list of files with revisions for which to collect those features. 
It saves the features to pickle, which can be used if needed.


#### 6. Training final classification model:
The training script for the best configuration of the model:
```commandline
python modules/train_model.py
```
In this section, we use the prepared data from previous sections. The sample of data used
can be found in **Prepared data** section. Full data will be published after the paper's publication. 

#### 7. Evaluation:
**Performance metrics:**
Model analysis based on performance metrics can be found in this notebook: 
[performance evaluation](https://github.com/trokhymovych/KI_multilingual_training/blob/main/notebooks/performance_metrics_calculation.ipynb)

(Note: This notebook also creates the input for fairness metrics)

**Fairness metrics:**
Model analysis based on fairness metrics can be found in this notebook: 
[fairness evaluation](https://github.com/trokhymovych/KI_multilingual_training/blob/main/notebooks/fairness_metrics_calculation.ipynb)

Also, we are using ORES scores as a reference. The script for their collection is: 
```commandline
python modules/ores_scores_collection.py
```
Or you can find the file with corresponding scores in the prepared data.

## Citation: 
**Fair Multilingual Vandalism Detection System for Wikipedia**
```
@inproceedings{10.1145/3580305.3599823,
author = {Trokhymovych, Mykola and Aslam, Muniza and Chou, Ai-Jou and Baeza-Yates, Ricardo and Saez-Trumper, Diego},
title = {Fair Multilingual Vandalism Detection System for Wikipedia},
year = {2023},
isbn = {9798400701030},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3580305.3599823},
doi = {10.1145/3580305.3599823},
abstract = {This paper presents a novel design of the system aimed at supporting the Wikipedia community in addressing vandalism on the platform. To achieve this, we collected a massive dataset of 47 languages, and applied advanced filtering and feature engineering techniques, including multilingual masked language modeling to build the training dataset from human-generated data. The performance of the system was evaluated through comparison with the one used in production in Wikipedia, known as ORES. Our research results in a significant increase in the number of languages covered, making Wikipedia patrolling more efficient to a wider range of communities. Furthermore, our model outperforms ORES, ensuring that the results provided are not only more accurate but also less biased against certain groups of contributors.},
booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4981–4990},
numpages = {10},
location = {Long Beach, CA, USA},
series = {KDD '23}
}
```
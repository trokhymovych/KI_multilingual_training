# Knowledge Integrity multilingual model training

This repository includes resources to reproduce training procedure for the paper
**Fair multilingual vandalism prevention system for Wikipedia** from data collection to model training. 

The model inference logic is implemented in 
[Knowledge Integrity repo](https://gitlab.wikimedia.org/repos/research/knowledge_integrity)


### Experiments reproducing:
#### 1. Data collection:
Data collection logic was done using Wikimedia analytical cluster. 
How to collect data on cluster:
```commandline
for i in ka lv ta ur eo lt sl hy hr sk eu et ms az da bg sr ro el th bn simple no hi ca hu ko fi vi uz sv cs he id tr uk nl pl ar fa it zh ru es ja de fr en
do
   python modules/data_collection.py -l $i -m train -f1 0 -f2 0 -n 300000 -uc 1 -p all_users
done
```
The script above collects dataset for each of given languages in the loop. 
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
For that purposes run:
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
Having collected training dataset we proceed with fine-tuning of MLMs. 
```commandline
python modules/feature_trainer.py
```

This script prepares the specific dataset for four different MLMs model tuning.
Later it uses prepared data and tune MLM for a pair of text classification (changes),
text classification (inserts, removes) and regression (title semantics). And later tune those 
models and save them to the \models directory.
GPU is needed for training. We used AMD Radeon Pro WX 9100 16GB GPU.


#### 4. MLM features calculation for datasets
Having collected training dataset and tuned MLMs we proceed with calculation text features for 
those datasets (training and testing parts)
```commandline
python modules/feature_builder.py
```
Before running the script we should make sure that paths to MLMs and datasets are correct in the script

#### 5. User features extraction for datasets
Additional user features extraction (is_anonymous, user group):
```commandline
python modules/users_features_collection.py
```
Module takes as an input the list of files with revisions for which to collect those features. 
It saves the features to pickle, so they can be used if needed.


#### 6. Training final classification model:
The training script for the best configuration of model:
```commandline
python modules/train_model.py
```
In this section we use the prepared data from previous sections. The sample of data used for training 
can be found later in **Prepared data** section. Full data will be published after the paper publication. 

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
Or you can find file with corresponding scores in the prepared data. 

### Prepared data: 
All processed data can be also found here: ...... (to be added)......





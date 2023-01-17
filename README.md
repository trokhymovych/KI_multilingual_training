# Knowledge Integrity multilingual model training

This repository includes resources to reproduce training procedure
of Knowledge Integrity multilingual model from data collection to model training. 

The model inference is implemented in 
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
It does not filter anonymous users (-f1) and revision wars (f2). 
Also, there is a limit of 300000 records per language.

Data collection of hold-out test:
```commandline
for i in ka lv ta ur eo lt sl hy hr sk eu et ms az da bg sr ro el th bn simple no hi ca hu ko fi vi uz sv cs he id tr uk nl pl ar fa it zh ru es ja de fr en
do
   python modules/data_collection.py -l $i -m test -f1 0 -f2 1 -n 100000 -uc 1 -p full_test
done
```

All collected data can be found here: .... (to be added)......


ToDo: add details of how to collect the data without cluster.

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
All processed data can be also found here: .... (to be added)......

#### 3. MLM tuning:
Having collected training dataset we proceed with fine-tuning of MLMs. 
```commandline
python modules/data_prepareration.py
```






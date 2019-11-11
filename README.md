# Graph-WaveNet_Modified

This is the modified version of original Graph-WaveNet model(https://github.com/nnzhan/Graph-WaveNet). \
The additional incident and weather features are included on top of the original model.

## Requirements
- python 3
- pytorch
- scipy
- numpy
- pandas
- pyaml
- pickle

## Data Preparation

### Step1:
 Creates dataframe using **create_dataframe.py**
 1. main_roads.csv => 주요도로 링크
```
python3 create_dataframe.py --data_path ./data/ --data_link ./main_roads.csv --save ./experiment/
```

### Step2:
 Creates train, val, test incident data through **incident_process.py** \
 **start_date and end_date_time** should be identical to original training data which are created by DCRNN scripts.
 ```
python3 incident_process.py --data ./data/ --adj_mx ./adj_mx.pkl --save ./experiment/ --start_date '5/1/2019' --end_date_time '5/31/2019 23:55:00'
```

## Experiment
 User needs to provide specific data path.
```
python3 train.py --incident true
```

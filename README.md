# Multitask Active Learning for Graph Anomaly Detection

## Requirements
This code requires the following:

- Python==3.8
- PyTorch==2.0.1
- Numpy==1.24.4
- DGL==1.1.1+cu102

## Usage
Take Cora dataset as an example:
```
python main.py  --dataset cora --strategy_ad medoids_spec_nent_diff --device 0 --alpha 1.25 --beta 0.5 --gamma 1 --cluster_num 24 --tau 0.95  --phi 1.25 
```
The hyperparameters for other datasets are reported in our paper.

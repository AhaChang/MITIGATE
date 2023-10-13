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
python main.py  --dataset cora --strategy_ad medoids_spec_nent_diff --result_path results/cora --device 0 --alpha $alpha --beta $beta --gamma 1 --cluster_num $c_num --weight_tmp $weight_tmp --max_budget $budget  --phi $phi 
```
The hyperparameters for other datasets are reported in our paper.

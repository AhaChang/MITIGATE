# Leveraging Active Learning with Auxiliary Task for Graph Anomaly Detection

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
The hyperparameters for other datasets are reported as follows.

|          | Cora | Citeseer | BlogCatalog | Flickr | Amazon | YelpChi |
|:--------:|:----:|:--------:|:-----------:|:------:|--------|---------|
|  $\tau$  | 0.95 |   0.90   |     0.98    |  0.98  | 0.98   | 0.985   |
| $\alpha$ | 1.25 |   0.50   |     1.25    |  1.25  | 1.25   | 0.5     |
|  $\beta$ | 0.50 |   2.00   |     1.00    |  0.50  | 0.8    | 1.25    |
|  $\phi$  | 1.25 |   2.00   |     1.00    |  0.50  | 10     | 8.0     |
|    $m$   |  24  |    24    |      18     |   27   | 10     | 20      |

The Amazon and YelpChi datasets can be found from [GADBench](https://github.com/squareRoot3/GADBench).

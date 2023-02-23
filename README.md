# TrustRD
Source code for SIGIR 2023 paper **Towards Trustworthy Rumor Detection with Interpretable Graph
Structural Learning**

## Dependencies
python 3.7    
pytorch 1.8.1   
pytorch_geometric 1.7.0 


## Usage

```
python ./Process/getTwittergraph.py Twitter15 # Encode graph for Twitter15
python ./Process/getTwittergraph.py Twitter16 # Encode graph for Twitter16
python ./Model/train.py Twitter15 100 # Run TrustRD for 100 iterations on Twitter15 dataset
python ./Model/train.py Twitter16 100 # Run TrustRD for 100 iterations on Twitter16 dataset
```

## Dataset
We use Twitter15 and Twitter16 dataset for the experiment.    
To learn more about the dataset, please refer to [RvNN](https://github.com/majingCUHK/Rumor_RvNN) for more details.

 






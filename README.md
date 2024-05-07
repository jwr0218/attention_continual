# cluster based memory stacking data  for Attention based model Continual Learning



<p align="center"><img src="https://github.com/jwr0218/attention_continual/assets/54136688/e615c923-2cf7-40d1-92c9-f092513fa6c4" width="750"/></p>
<p align="center"><img src="https://github.com/jwr0218/attention_continual/assets/54136688/db7d036d-e1da-435e-9f0f-de364e208638" width="750"/></p>

## Data Preparation

Datasets are used in the following file structure:

```
│continual_learning for MIL/
├──data/
│  ├── AMI
│  │   ├── month_continual
│  │   │   ├──  summer
│  │   │   ├──  winter
│  │   │   ├──  all
│  ├── ETRI
│  │   ├── date_continual
│  │   │   ├──  pickle_0
│  │   │   ├──  pickle_10000.pkl
│  │   │   ├──  pickle_20000.pkl
│  │   │   ├──  pickle_30000.pkl
│  │   │   ├──  pickle_40000.pkl
│  │   │   ├──  pickle_50000.pkl
│  │   │   ├──  pickle_60000.pkl
│  │   │   ├──  pickle_70000.pkl
│  │   │   ├──  pickle_80000.pkl
│  │   │   ├──  pickle_90000.pkl
```


### AMI Dataset
Time Series Regression (Elec, water, gas hotwater, hot) ( classified ) 

  
### ETRI Human Life log Dataset
you can download it from [here](https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR)


## Setups

All code was developed and tested on Nvidia RTX A4000 (48SMs, 16GB) the following environment.
- Ubuntu 18.04
- python 3.10.11
- torch 2.0.1
- numpy 1.24.3
- pandas 2.1.0
- scikit-learn 1.3.0
- scipy 1.11.2

## Implement

```shell
python AMI_continual_cluster.py (summer/winter/all)
ex) python AMI_continual_cluster.py summer 

python ETRI_continual_cluster.py 

```

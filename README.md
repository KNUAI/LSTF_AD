# LSTF_AD
Long Sequence Time-Series Forecasting & Anomlay Detection

## Environments
GPU: NVIDIA TITAN RTX 24GB
python 3.8

## Installation
```
git clone https://github.com/KNUAI/LSTF_AD.git
```
```
cd LSTF_AD && pip install -r requirements.txt
```

If you want to LSTF_AD, put your data in ./SKAB/data
If you want to sample data for LSTF_AD, you can download this [SKAB](https://github.com/waico/SKAB)  
```
git clone https://github.com/waico/SKAB.git
```

## Usage
**If you want to train and test LSTF_AD(unsupervised) at once**
```
python SKAB_LSTFAD.py --model MCF --ad_r_model LSTM
```

Write the model you want to use in model!  
Write the ad_r_model you want to use in ad_r_model!  

**If you want to train and test LSTF_AD(supervised) at once**
```
python SKAB_LSTFAD2.py --model MCF --ad_model DNN
```

Write the model you want to use in model!  
Write the ad_model you want to use in ad_model!  

## Results
**If you finish the experiment, you can see experiment results in picture directory**
**Also, you can see experiments_log.txt**



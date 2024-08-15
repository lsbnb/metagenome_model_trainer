# Metagenome Model Trainer
A machine learning model for metagenome data training.
![Logo_ccs_model_trainer](https://github.com/lsbnb/metagenome_model_trainer/assets/51230850/d9f25911-4574-4a27-8e2a-1edd56dcf3e7)

#### Contact information: 
Chung-Yen Lin (cylin@iis.sinica.edu.tw); [**LAB website**](http://eln.iis.sinica.edu.tw)

## Usage:

There are two functions in Metagenome Model Trainer script, (1.) Training built-in metagenomics data which from protein sequences transformed to Enzyme Category (EC) numbers by CLEAN. (2.) Training sample data sample_data/ccs_sample.csv

#### 1. Training built-in metagenomics data:
```
  python metagenome_model_trainer.py
```

#### 2. Training sample data:
```
  python metagenome_model_trainer.py sample_data/sample_pattern1.csv
```

## Result:

#### Pattern 1:
Accuracy: 0.64	Specificity: 0.56	MCC: 0.30	F1: 0.58

#### Pattern 2:
Accuracy: 0.92	Specificity: 0.94	MCC: 0.85	F1: 0.92

#### Pattern 3:
Accuracy: 0.94	Specificity: 0.98	MCC: 0.89	F1: 0.94

#### Pattern 4:
Accuracy: 0.96	Specificity: 0.94	MCC: 0.92	F1: 0.95

#### Pattern 5:
Accuracy: 0.93	Specificity: 0.94	MCC: 0.85	F1: 0.92

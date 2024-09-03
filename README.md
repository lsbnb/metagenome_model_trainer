# Metagenome Model Trainer
A machine learning model for metagenome data training.
![Logo_ccs_model_trainer](https://github.com/lsbnb/metagenome_model_trainer/assets/51230850/d9f25911-4574-4a27-8e2a-1edd56dcf3e7)

#### Contact information: 
Chung-Yen Lin (cylin@iis.sinica.edu.tw); [**LAB website**](http://eln.iis.sinica.edu.tw)

## Environment:

#### Python 3.9.13
#### SHAP v0.42.1
#### Pandas v1.4.4
#### matplotlib v3.5.2

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
#### Accuracy: 0.64	Specificity: 0.56	MCC: 0.30	F1: 0.58
![ccs_pattern1_dot](https://github.com/user-attachments/assets/235494b8-f592-40e7-ade2-abaf103d9866)

#### Pattern 2:
#### Accuracy: 0.92	Specificity: 0.94	MCC: 0.85	F1: 0.92
![ccs_pattern2_dot](https://github.com/user-attachments/assets/d239890a-0118-4057-a55d-12d4d9ab791b)

#### Pattern 3:
#### Accuracy: 0.94	Specificity: 0.98	MCC: 0.89	F1: 0.94
![ccs_pattern3_dot](https://github.com/user-attachments/assets/8930dc2a-ec7c-42c2-b3f0-2668ff4fc4a5)

#### Pattern 4:
#### Accuracy: 0.96	Specificity: 0.94	MCC: 0.92	F1: 0.95
![ccs_pattern4_dot](https://github.com/user-attachments/assets/ab437b26-34f0-4868-90df-7bccc4aaebac)

#### Pattern 5:
#### Accuracy: 0.93	Specificity: 0.94	MCC: 0.85	F1: 0.92
![ccs_pattern5_dot](https://github.com/user-attachments/assets/af8c0b77-4d05-4d7e-9530-782fa12546d5)

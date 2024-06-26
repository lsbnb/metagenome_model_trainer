import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import ensemble, preprocessing, metrics
from sklearn import datasets
import sys



if len(sys.argv) >= 2:
	print('Training...')
	hbcd_deg_train_test = pd.read_csv(sys.argv[1])
	hbcd_deg_X_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[2:8227]].values
	hbcd_deg_y_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[1]].values

	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1)
	forest = ensemble.RandomForestClassifier(n_estimators = 100)
	forest_fit = forest.fit(train_X, train_y)
	test_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, test_y_predicted)
	T1_mcc=metrics.matthews_corrcoef(test_y, test_y_predicted)
	T1_f1 = metrics.f1_score(test_y, test_y_predicted)
	total1=sum(sum(cm1))
	T1_accuracy=(cm1[0,0]+cm1[1,1])/total1
	T1_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	print('Pattern 1:')
	print('Accuracy: '+str(T1_accuracy)+'\t'+'Specificity: '+str(T1_specificity)+'\t'+'MCC: '+str(T1_mcc)+'\t'+'F1: '+str(T1_f1))

else:
	print('Build-in model training...')
	url1 = "https://eln.iis.sinica.edu.tw/lims/files/users/ph/hbcd_deg/CCS/clean/clean_ccs_bc.csv"
	hbcd_deg_train_g1 = pd.read_csv(url1)
	hbcd_deg_X_g1 = hbcd_deg_train_g1[hbcd_deg_train_g1.columns[2:8227]].values
	hbcd_deg_y_g1 = hbcd_deg_train_g1[hbcd_deg_train_g1.columns[1]].values

	url2 = "https://eln.iis.sinica.edu.tw/lims/files/users/ph/hbcd_deg/CCS/clean/clean_ccs_bs.csv"
	hbcd_deg_train_g2 = pd.read_csv(url2)
	hbcd_deg_X_g2 = hbcd_deg_train_g2[hbcd_deg_train_g2.columns[2:8227]].values
	hbcd_deg_y_g2 = hbcd_deg_train_g2[hbcd_deg_train_g2.columns[1]].values

	url3 = "https://eln.iis.sinica.edu.tw/lims/files/users/ph/hbcd_deg/CCS/clean/clean_ccs_dm.csv"
	hbcd_deg_train_g3 = pd.read_csv(url3)
	hbcd_deg_X_g3 = hbcd_deg_train_g3[hbcd_deg_train_g3.columns[2:8227]].values
	hbcd_deg_y_g3 = hbcd_deg_train_g3[hbcd_deg_train_g3.columns[1]].values

	url4 = "https://eln.iis.sinica.edu.tw/lims/files/users/ph/hbcd_deg/CCS/clean/clean_ccs_pa.csv"
	hbcd_deg_train_g4 = pd.read_csv(url4)
	hbcd_deg_X_g4 = hbcd_deg_train_g4[hbcd_deg_train_g4.columns[2:8227]].values
	hbcd_deg_y_g4 = hbcd_deg_train_g4[hbcd_deg_train_g4.columns[1]].values

	url5 = "https://eln.iis.sinica.edu.tw/lims/files/users/ph/hbcd_deg/CCS/clean/clean_ccs_rp.csv"
	hbcd_deg_train_g5 = pd.read_csv(url5)
	hbcd_deg_X_g5 = hbcd_deg_train_g5[hbcd_deg_train_g5.columns[2:8227]].values
	hbcd_deg_y_g5 = hbcd_deg_train_g5[hbcd_deg_train_g5.columns[1]].values

	#Group1_vaild_RM
	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g1, hbcd_deg_y_g1, test_size = 0.1, random_state=25)
	forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=6)
	forest_fit = forest.fit(train_X, train_y)
	G1_valid_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, G1_valid_y_predicted)
	G1_mcc=metrics.matthews_corrcoef(test_y, G1_valid_y_predicted)
	G1_f1 = metrics.f1_score(test_y, G1_valid_y_predicted)
	total1=sum(sum(cm1))
	G1_accuracy=(cm1[0,0]+cm1[1,1])/total1
	G1_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	#Group2_vaild_RM
	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g2, hbcd_deg_y_g2, test_size = 0.1, random_state=47)
	forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=21)
	forest_fit = forest.fit(train_X, train_y)
	G2_valid_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, G2_valid_y_predicted)
	G2_mcc=metrics.matthews_corrcoef(test_y, G2_valid_y_predicted)
	G2_f1 = metrics.f1_score(test_y, G2_valid_y_predicted)
	total1=sum(sum(cm1))
	G2_accuracy=(cm1[0,0]+cm1[1,1])/total1
	G2_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	#Group3_vaild_RM
	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g3, hbcd_deg_y_g3, test_size = 0.1, random_state=2)
	forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=7)
	forest_fit = forest.fit(train_X, train_y)
	G3_valid_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, G3_valid_y_predicted)
	G3_mcc=metrics.matthews_corrcoef(test_y, G3_valid_y_predicted)
	G3_f1 = metrics.f1_score(test_y, G3_valid_y_predicted)
	total1=sum(sum(cm1))
	G3_accuracy=(cm1[0,0]+cm1[1,1])/total1
	G3_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	#Group4_vaild_RM
	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g4, hbcd_deg_y_g4, test_size = 0.1, random_state=1)
	forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=16)
	forest_fit = forest.fit(train_X, train_y)
	G4_valid_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, G4_valid_y_predicted)
	G4_mcc=metrics.matthews_corrcoef(test_y, G4_valid_y_predicted)
	G4_f1 = metrics.f1_score(test_y, G4_valid_y_predicted)
	total1=sum(sum(cm1))
	G4_accuracy=(cm1[0,0]+cm1[1,1])/total1
	G4_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	#Group5_vaild_RM
	train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g5, hbcd_deg_y_g5, test_size = 0.1, random_state=8)
	forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=0)
	forest_fit = forest.fit(train_X, train_y)
	G5_valid_y_predicted = forest.predict(test_X)
	cm1 = metrics.confusion_matrix(test_y, G5_valid_y_predicted)
	G5_mcc=metrics.matthews_corrcoef(test_y, G5_valid_y_predicted)
	G5_f1 = metrics.f1_score(test_y, G5_valid_y_predicted)
	total1=sum(sum(cm1))
	G5_accuracy=(cm1[0,0]+cm1[1,1])/total1
	G5_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	print('Pattern 1:')
	print('Accuracy: '+str(G1_accuracy)+'\t'+'Specificity: '+str(G1_specificity)+'\t'+'MCC: '+str(G1_mcc)+'\t'+'F1: '+str(G1_f1))
	print('Pattern 2:')
	print('Accuracy: '+str(G2_accuracy)+'\t'+'Specificity: '+str(G2_specificity)+'\t'+'MCC: '+str(G2_mcc)+'\t'+'F1: '+str(G2_f1))
	print('Pattern 3:')
	print('Accuracy: '+str(G3_accuracy)+'\t'+'Specificity: '+str(G3_specificity)+'\t'+'MCC: '+str(G3_mcc)+'\t'+'F1: '+str(G3_f1))
	print('Pattern 4:')
	print('Accuracy: '+str(G4_accuracy)+'\t'+'Specificity: '+str(G4_specificity)+'\t'+'MCC: '+str(G4_mcc)+'\t'+'F1: '+str(G4_f1))
	print('Pattern 5:')
	print('Accuracy: '+str(G5_accuracy)+'\t'+'Specificity: '+str(G5_specificity)+'\t'+'MCC: '+str(G5_mcc)+'\t'+'F1: '+str(G5_f1))


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
	print('Training input sample...It takes about 2 minutes...')
	hbcd_deg_train_test = pd.read_csv(sys.argv[1])
	hbcd_deg_X_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[2:8227]].values
	hbcd_deg_y_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[1]].values

	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1)
		i=i+1
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		test_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, test_y_predicted)
		T1_mcc=metrics.matthews_corrcoef(test_y, test_y_predicted)
		T1_f1 = metrics.f1_score(test_y, test_y_predicted)
		total1=sum(sum(cm1))
		T1_accuracy=(cm1[0,0]+cm1[1,1])/total1
		T1_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(T1_accuracy)
		sp_list.append(T1_specificity)
		mcc_list.append(T1_mcc)
		f1_list.append(T1_f1)

	T1_RM_acc_mean = statistics.fmean(acc_list)
	T1_RM_sp_mean = statistics.fmean(sp_list)
	T1_RM_mcc_mean = statistics.fmean(mcc_list)
	T1_RM_f1_mean = statistics.fmean(f1_list)
	print('Pattern training result:')
	print('Accuracy: %.2f' % T1_RM_acc_mean +'\t'+'Specificity: %.2f' % T1_RM_sp_mean +'\t'+'MCC: %.2f' % T1_RM_mcc_mean +'\t'+'F1: %.2f' % T1_RM_f1_mean)

else:
	print('If the script is interrupted...Please try again...')
	print('Build-in model training...It takes about 10 minutes...')
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
	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		i=i+1
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g1, hbcd_deg_y_g1, test_size = 0.1)
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		G1_valid_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, G1_valid_y_predicted)
		G1_mcc=metrics.matthews_corrcoef(test_y, G1_valid_y_predicted)
		G1_f1 = metrics.f1_score(test_y, G1_valid_y_predicted)
		total1=sum(sum(cm1))
		G1_accuracy=(cm1[0,0]+cm1[1,1])/total1
		G1_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(G1_accuracy)
		sp_list.append(G1_specificity)
		mcc_list.append(G1_mcc)
		f1_list.append(G1_f1)
	G1_RM_acc_mean = statistics.fmean(acc_list)
	G1_RM_sp_mean = statistics.fmean(sp_list)
	G1_RM_mcc_mean = statistics.fmean(mcc_list)
	G1_RM_f1_mean = statistics.fmean(f1_list)

	#Group2_vaild_RM
	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		i=i+1
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g2, hbcd_deg_y_g2, test_size = 0.1)
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		G2_valid_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, G2_valid_y_predicted)
		G2_mcc=metrics.matthews_corrcoef(test_y, G2_valid_y_predicted)
		G2_f1 = metrics.f1_score(test_y, G2_valid_y_predicted)
		total1=sum(sum(cm1))
		G2_accuracy=(cm1[0,0]+cm1[1,1])/total1
		G2_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(G2_accuracy)
		sp_list.append(G2_specificity)
		mcc_list.append(G2_mcc)
		f1_list.append(G2_f1)
	G2_RM_acc_mean = statistics.fmean(acc_list)
	G2_RM_sp_mean = statistics.fmean(sp_list)
	G2_RM_mcc_mean = statistics.fmean(mcc_list)
	G2_RM_f1_mean = statistics.fmean(f1_list)

	#Group3_vaild_RM
	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		i=i+1
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g3, hbcd_deg_y_g3, test_size = 0.1)
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		G3_valid_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, G3_valid_y_predicted)
		G3_mcc=metrics.matthews_corrcoef(test_y, G3_valid_y_predicted)
		G3_f1 = metrics.f1_score(test_y, G3_valid_y_predicted)
		total1=sum(sum(cm1))
		G3_accuracy=(cm1[0,0]+cm1[1,1])/total1
		G3_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(G3_accuracy)
		sp_list.append(G3_specificity)
		mcc_list.append(G3_mcc)
		f1_list.append(G3_f1)
	G3_RM_acc_mean = statistics.fmean(acc_list)
	G3_RM_sp_mean = statistics.fmean(sp_list)
	G3_RM_mcc_mean = statistics.fmean(mcc_list)
	G3_RM_f1_mean = statistics.fmean(f1_list)

	#Group4_vaild_RM
	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		i=i+1
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g4, hbcd_deg_y_g4, test_size = 0.1)
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		G4_valid_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, G4_valid_y_predicted)
		G4_mcc=metrics.matthews_corrcoef(test_y, G4_valid_y_predicted)
		G4_f1 = metrics.f1_score(test_y, G4_valid_y_predicted)
		total1=sum(sum(cm1))
		G4_accuracy=(cm1[0,0]+cm1[1,1])/total1
		G4_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(G4_accuracy)
		sp_list.append(G4_specificity)
		mcc_list.append(G4_mcc)
		f1_list.append(G4_f1)
	G4_RM_acc_mean = statistics.fmean(acc_list)
	G4_RM_sp_mean = statistics.fmean(sp_list)
	G4_RM_mcc_mean = statistics.fmean(mcc_list)
	G4_RM_f1_mean = statistics.fmean(f1_list)

	#Group5_vaild_RM
	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 300 :
		i=i+1
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_g5, hbcd_deg_y_g5, test_size = 0.1)
		forest = ensemble.RandomForestClassifier(n_estimators = 100)
		forest_fit = forest.fit(train_X, train_y)
		G5_valid_y_predicted = forest.predict(test_X)
		cm1 = metrics.confusion_matrix(test_y, G5_valid_y_predicted)
		G5_mcc=metrics.matthews_corrcoef(test_y, G5_valid_y_predicted)
		G5_f1 = metrics.f1_score(test_y, G5_valid_y_predicted)
		total1=sum(sum(cm1))
		G5_accuracy=(cm1[0,0]+cm1[1,1])/total1
		G5_specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
		acc_list.append(G5_accuracy)
		sp_list.append(G5_specificity)
		mcc_list.append(G5_mcc)
		f1_list.append(G5_f1)
	G5_RM_acc_mean = statistics.fmean(acc_list)
	G5_RM_sp_mean = statistics.fmean(sp_list)
	G5_RM_mcc_mean = statistics.fmean(mcc_list)
	G5_RM_f1_mean = statistics.fmean(f1_list)

	print('Pattern 1:')
	print('Accuracy: %.2f' % G1_RM_acc_mean +'\t'+'Specificity: %.2f' % G1_RM_sp_mean +'\t'+'MCC: %.2f' % G1_RM_mcc_mean +'\t'+'F1: %.2f' % G1_RM_f1_mean)
	print('Pattern 2:')
	print('Accuracy: %.2f' % G2_RM_acc_mean +'\t'+'Specificity: %.2f' % G2_RM_sp_mean +'\t'+'MCC: %.2f' % G2_RM_mcc_mean +'\t'+'F1: %.2f' % G2_RM_f1_mean)
	print('Pattern 3:')
	print('Accuracy: %.2f' % G3_RM_acc_mean +'\t'+'Specificity: %.2f' % G3_RM_sp_mean +'\t'+'MCC: %.2f' % G3_RM_mcc_mean +'\t'+'F1: %.2f' % G3_RM_f1_mean)
	print('Pattern 4:')
	print('Accuracy: %.2f' % G4_RM_acc_mean +'\t'+'Specificity: %.2f' % G4_RM_sp_mean +'\t'+'MCC: %.2f' % G4_RM_mcc_mean +'\t'+'F1: %.2f' % G4_RM_f1_mean)
	print('Pattern 5:')
	print('Accuracy: %.2f' % G5_RM_acc_mean +'\t'+'Specificity: %.2f' % G5_RM_sp_mean +'\t'+'MCC: %.2f' % G5_RM_mcc_mean +'\t'+'F1: %.2f' % G5_RM_f1_mean)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import ensemble, preprocessing, metrics
from sklearn import datasets
import sys
import shap



if len(sys.argv) >= 2:
	print('Training input sample...It takes about 2 minutes...')
	hbcd_deg_train_test = pd.read_csv(sys.argv[1])
	hbcd_deg_X_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[2:8227]].values
	hbcd_deg_y_test = hbcd_deg_train_test[hbcd_deg_train_test.columns[1]].values
	feature_list = hbcd_deg_train_test.columns[2:8228]

	i = 0
	acc_list = []
	sp_list = []
	mcc_list = []
	f1_list = []

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_test):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_test[train_X_idx], hbcd_deg_y_test[train_X_idx])
			test_y=hbcd_deg_y_test[test_X_idx]
			G1_test_y_predicted = forest.predict(hbcd_deg_X_test[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G1_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G1_test_y_predicted)
			f1 = metrics.f1_score(test_y, G1_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
	T1_RM_acc_mean = statistics.fmean(acc_list)
	T1_RM_sp_mean = statistics.fmean(sp_list)
	T1_RM_mcc_mean = statistics.fmean(mcc_list)
	T1_RM_f1_mean = statistics.fmean(f1_list)
	print('Pattern training result:')
	print('Accuracy: %.2f' % T1_RM_acc_mean +'\t'+'Specificity: %.2f' % T1_RM_sp_mean +'\t'+'MCC: %.2f' % T1_RM_mcc_mean +'\t'+'F1: %.2f' % T1_RM_f1_mean)

	print('SHAP analysis...')
	P_name_idx = sys.argv[1].find('sample_pattern')
	if P_name_idx == -1:
		train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=0)
		forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=0)
		explainer = shap.Explainer(forest)
		shap_values = explainer.shap_values(test_X)
		plt.clf()
		plt.title('RandomForest SHAP analysis',fontsize=15,fontproperties="Times New Roman",weight='bold')
		shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)
	else:
		P_name = sys.argv[1].split('.')[0].split('pattern')[1]
		if P_name == '1':
			train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=25)
			forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=6)
			explainer = shap.Explainer(forest)
			shap_values = explainer.shap_values(test_X)
			plt.clf()
			plt.title('CCS Pattern 1 : RandomForest',fontsize=15,fontproperties="Times New Roman",weight='bold')
			shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)
		elif P_name == '2':
			train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=47)
			forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=21)
			explainer = shap.Explainer(forest)
			shap_values = explainer.shap_values(test_X)
			plt.clf()
			plt.title('CCS Pattern 2 : RandomForest',fontsize=15,fontproperties="Times New Roman",weight='bold')
			shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)
		elif P_name == '3':
			train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=2)
			forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=7)
			explainer = shap.Explainer(forest)
			shap_values = explainer.shap_values(test_X)
			plt.clf()
			plt.title('CCS Pattern 3 : RandomForest',fontsize=15,fontproperties="Times New Roman",weight='bold')
			shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)
		elif P_name == '4':
			train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=1)
			forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=16)
			explainer = shap.Explainer(forest)
			shap_values = explainer.shap_values(test_X)
			plt.clf()
			plt.title('CCS Pattern 4 : RandomForest',fontsize=15,fontproperties="Times New Roman",weight='bold')
			shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)
		elif P_name == '5':
			train_X, test_X, train_y, test_y = train_test_split(hbcd_deg_X_test, hbcd_deg_y_test, test_size = 0.1, random_state=8)
			forest = ensemble.RandomForestClassifier(n_estimators = 100, random_state=0)
			explainer = shap.Explainer(forest)
			shap_values = explainer.shap_values(test_X)
			plt.clf()
			plt.title('CCS Pattern 5 : RandomForest',fontsize=15,fontproperties="Times New Roman",weight='bold')
			shap.summary_plot(shap_values[0], test_X, feature_names=feature_list, max_display=15)

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

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_g1):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_g1[train_X_idx], hbcd_deg_y_g1[train_X_idx])
			test_y=hbcd_deg_y_g1[test_X_idx]
			G1_test_y_predicted = forest.predict(hbcd_deg_X_g1[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G1_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G1_test_y_predicted)
			f1 = metrics.f1_score(test_y, G1_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
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

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_g2):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_g2[train_X_idx], hbcd_deg_y_g2[train_X_idx])
			test_y=hbcd_deg_y_g2[test_X_idx]
			G2_test_y_predicted = forest.predict(hbcd_deg_X_g2[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G2_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G2_test_y_predicted)
			f1 = metrics.f1_score(test_y, G2_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
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

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_g3):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_g3[train_X_idx], hbcd_deg_y_g3[train_X_idx])
			test_y=hbcd_deg_y_g3[test_X_idx]
			G3_test_y_predicted = forest.predict(hbcd_deg_X_g3[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G3_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G3_test_y_predicted)
			f1 = metrics.f1_score(test_y, G3_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
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

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_g4):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_g4[train_X_idx], hbcd_deg_y_g4[train_X_idx])
			test_y=hbcd_deg_y_g4[test_X_idx]
			G4_test_y_predicted = forest.predict(hbcd_deg_X_g4[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G4_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G4_test_y_predicted)
			f1 = metrics.f1_score(test_y, G4_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
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

	while i < 30 :
		kf = KFold(n_splits=10, shuffle=True)
		i=i+1
		for train_X_idx, test_X_idx in kf.split(hbcd_deg_X_g5):
			forest = ensemble.RandomForestClassifier(n_estimators = 100)
			forest_fit = forest.fit(hbcd_deg_X_g5[train_X_idx], hbcd_deg_y_g5[train_X_idx])
			test_y=hbcd_deg_y_g5[test_X_idx]
			G5_test_y_predicted = forest.predict(hbcd_deg_X_g5[test_X_idx])
			cm1 = metrics.confusion_matrix(test_y, G5_test_y_predicted)
			mcc=metrics.matthews_corrcoef(test_y, G5_test_y_predicted)
			f1 = metrics.f1_score(test_y, G5_test_y_predicted)
			total1=sum(sum(cm1))
			accuracy=(cm1[0,0]+cm1[1,1])/total1
			specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
			acc_list.append(accuracy)
			sp_list.append(specificity)
			mcc_list.append(mcc)
			f1_list.append(f1)
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


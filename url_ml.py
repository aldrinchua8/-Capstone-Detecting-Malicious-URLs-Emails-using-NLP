
#https://thesai.org/Downloads/Volume11No1/Paper_19-Malicious_URL_Detection_based_on_Machine_Learning.pdf


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve
from scipy.special import expit
import statsmodels.api as sm

def ConfMatrixDisp(y_test, predictions, label, name, title):
    cm = confusion_matrix(y_test, predictions, labels=label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot()
    plt.title(title + " URL Confusion Matrix")
    #plt.show()
    plt.savefig(name + '.png', dpi=100)
    plt.clf()


df = pd.read_csv('urltst.csv')
df = df.dropna()




# print(df.head(10))
# print(df.columns)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values



X_train, X_init, y_train, y_init = train_test_split(X, y, test_size = 0.30, random_state = 0)

X_test, X_valid, y_test, y_valid = train_test_split(X_init, y_init, test_size = 0.50, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_valid = sc.transform(X_valid)


######################################################################################################################################
models = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Trees', 'Ensemble']
df_results = pd.DataFrame({'Model': models, 'Accuracy': models, 'Precision': models, 
                            'Sensitivity': models, 'Specificity':models})


df_results = df_results.set_index(['Model'])
print(df_results)

###################################################################################################################################
SVM = SVC(kernel = 'linear', random_state = 0)

SVM.fit(X_train, y_train)

y_pred = SVM.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)



tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("SVM")
print(conf_matrix)
ConfMatrixDisp(y_test, y_pred, SVM.classes_, 'NaiveBayes', 'Naive Bayes')
accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
df_results.loc['SVM', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['SVM', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['SVM', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['SVM', 'Specificity'] = specificity

#y_pred_proba = SVM.predict_proba(X_test)[::, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test,  y_pred)
auc_svm = roc_auc_score(y_test, y_pred)
plt.plot(fpr_svm,tpr_svm,label="AUC="+str(auc_svm))
plt.title('ROC Curve for SVM')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_SVM.png')
plt.clf()

##########################################################################################################################
LR = LogisticRegression()

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)


tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("LR")
print(conf_matrix)
ConfMatrixDisp(y_test, y_pred, LR.classes_, 'LogReg', 'Logistic Regression')
print(accuracy_score(y_test, y_pred))
df_results.loc['Logistic Regression', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['Logistic Regression', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['Logistic Regression', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['Logistic Regression', 'Specificity'] = specificity

#y_pred_proba = LR.predict_proba(X_test)[::, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test,  y_pred)
auc_lr = roc_auc_score(y_test, y_pred)
plt.plot(fpr_lr,tpr_lr,label="AUC="+str(auc_lr))
plt.title('ROC Curve for Logistic Regression')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_LR.png')
plt.clf()

##########################################################################################################################
NB = GaussianNB()

NB.fit(X_train, y_train)

y_pred = NB.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)


tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("NB")
print(conf_matrix)
ConfMatrixDisp(y_test, y_pred, NB.classes_, 'NaiveBayes', 'Naive Bayes')
print(accuracy_score(y_test, y_pred))
df_results.loc['Naive Bayes', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['Naive Bayes', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['Naive Bayes', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['Naive Bayes', 'Specificity'] = specificity

#y_pred_proba = NB.predict_proba(X_test)[::, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test,  y_pred)
auc_nb = roc_auc_score(y_test, y_pred)
plt.plot(fpr_nb,tpr_nb,label="AUC="+str(auc_nb))
plt.title('ROC Curve for Naive Bayes')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_NB.png')
plt.clf()


##########################################################################################################################
DT = DecisionTreeClassifier(max_depth = 3)

DT.fit(X_train, y_train)


y_pred = DT.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("DT")
print(conf_matrix)
ConfMatrixDisp(y_test, y_pred, DT.classes_, 'DTree', 'Decision Tree')
print(accuracy_score(y_test, y_pred))
df_results.loc['Decision Trees', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['Decision Trees', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['Decision Trees', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['Decision Trees', 'Specificity'] = specificity

df_results.to_excel('Initialresults.xlsx')

# ccp_alpha = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 0.10,0.11,0.12,0.13,0.14, 0.15, 0.20, 0.25]
# train_scores, test_scores = validation_curve(
#     DT,X, y, param_name="ccp_alpha", param_range=ccp_alpha, scoring="neg_mean_absolute_error", n_jobs=2)
# train_errors, test_errors = -train_scores, -test_scores

# plt.plot(ccp_alpha, train_errors.mean(axis=1), label="Training error")
# plt.plot(ccp_alpha, test_errors.mean(axis=1), label="Testing error")
# plt.legend()

path = DT.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']
print(alphas)

accuracy_train, accuracy_test = [], []

for i in alphas:
    DT1 = DecisionTreeClassifier(ccp_alpha = i)

    DT1.fit(X_train, y_train)
    y_train_pred = DT1.predict(X_train)
    y_test_pred = DT1.predict(X_test)

    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

plt.plot(alphas,accuracy_train, label = "Train Accuracy")
plt.plot(alphas,accuracy_test, label = "Test Accuracy")
plt.savefig('validation.png')
plt.clf()

# train_scores, test_scores = validation_curve(
#     DT,X, y, param_name="ccp_alpha", param_range=alphas, scoring="neg_mean_absolute_error", n_jobs=2)
# train_errors, test_errors = -train_scores, -test_scores

# plt.plot(alphas, train_errors.mean(axis=1), label="Training error")
# plt.plot(alphas, test_errors.mean(axis=1), label="Testing error")
# plt.legend()
# plt.xlabel("Maximum depth of decision tree")
# plt.ylabel("Mean absolute error (k$)")
# _ = plt.title("Validation curve for decision tree")
# plt.savefig('validation23.png')
# plt.clf()
# #y_pred_proba = DT.predict_proba(X_test)[::, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test,  y_pred)
auc_dt = roc_auc_score(y_test, y_pred)
plt.plot(fpr_dt,tpr_dt,label="AUC="+str(auc_dt))
plt.title('ROC Curve for Decision Tree')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_DT.png')
plt.clf()

###########################################################################################################################
Ensamble = VotingClassifier([('LR', LR), ('SVM', SVM), ('DT', DT)], voting = 'hard')

Ensamble.fit(X_train, y_train)

y_pred = Ensamble.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("ENS")
print(conf_matrix)
print(accuracy_score(y_test, y_pred))

df_results.loc['Ensemble', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['Ensemble', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['Ensemble', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['Ensemble', 'Specificity'] = specificity


fpr_ens, tpr_ens, _ = roc_curve(y_test,  y_pred)
auc_ens = roc_auc_score(y_test, y_pred)
plt.plot(fpr_ens,tpr_ens,label="AUC="+str(auc_ens))
plt.title('ROC Curve for Ensemble')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_Ensemble.png')
plt.clf()

###################################################################################################
#DT is my chosen model

df_chosen = pd.DataFrame({'Model': ['Ensemble'], 'Accuracy': ['Ensemble'], 'Precision': ['Ensemble'], 
                            'Sensitivity': ['Ensemble'], 'Specificity':['Ensemble']})

df_chosen = df_chosen.set_index(['Model'])
y_pred = DT.predict(X_valid)

conf_matrix = confusion_matrix(y_valid, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("Ensemble")
print(conf_matrix)
print(accuracy_score(y_valid, y_pred))

df_chosen.loc['Ensemble', 'Accuracy'] = accuracy_score(y_valid, y_pred)

print(precision_score(y_valid, y_pred))
df_chosen.loc['Ensemble', 'Precision'] = precision_score(y_valid, y_pred)

print(recall_score(y_valid, y_pred))
df_chosen.loc['Ensemble', 'Sensitivity'] = recall_score(y_valid, y_pred)

print(specificity)
df_chosen.loc['Ensemble', 'Specificity'] = specificity

#y_pred_proba = Ensamble.predict_proba(X_test)[::, 1]
fpr_dtch, tpr_dtch, _ = roc_curve(y_valid,  y_pred)
auc_dtch = roc_auc_score(y_valid, y_pred)
plt.plot(fpr_dtch,tpr_dtch,label="AUC="+str(auc_dtch))
plt.title('ROC Curve for Decision Tree')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_Ensemble_chosen.png')
plt.clf()

df_chosen.to_excel('URLChosen.xlsx')

fig, ax = plt.subplots(figsize = (6,6))
# import matplotlib.patheffects as mpe
# outline=mpe.withStroke(linewidth=6, foreground='black')
#NB
# plt.plot(fpr_nb, tpr_nb, color = "black", linewidth = 6)
plt.plot(fpr_nb, tpr_nb, label = "Naive Bayes AUC=" + str(auc_nb.round(4)), color = "purple", linewidth = 4, alpha = 0.5)

# plt.plot(fpr_lr, tpr_lr, label = "Log. Regression AUC=" + str(auc_lr.round(4)), color = "red", linewidth = 4, alpha = 0.5, path_effects = [outline])
plt.plot(fpr_svm, tpr_svm, label = "SVM AUC=" + str(auc_svm.round(4)), color = "blue", linewidth = 4, alpha = 0.5)
plt.plot(fpr_dt, tpr_dt, label = "Dec. Trees AUC=" + str(auc_dt.round(4)), color = "orange", linewidth = 4, alpha = 0.5)
plt.plot(fpr_ens, tpr_ens, label = "Ensemble AUC=" + str(auc_ens.round(4)), color = "green", linewidth = 4, alpha = 0.5)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for URL Models")
plt.legend()
plt.savefig("AllROCUrl.png")


df_results.to_excel('UrlInitialresults.xlsx')
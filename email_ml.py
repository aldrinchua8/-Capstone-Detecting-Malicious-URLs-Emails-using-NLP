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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve
from scipy.special import expit
import statsmodels.api as sm

stop_words = nltk.corpus.stopwords.words('english')

stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)

def ConfMatrixDisp(y_test, predictions, label, name, title):
    cm = confusion_matrix(y_test, predictions, labels=label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot()
    plt.title(title + " Confusion Matrix")
    #plt.show()
    plt.savefig(name + '.png', dpi=100)
    plt.clf()

#print(stop_words)
df_email = pd.read_excel('clean_email.xlsx')
df_email = df_email[['text', 'spam']]
df_email['text'] = df_email['text'].str.replace(r'[!\"#\＄%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]', " ")

df_email['text'] = df_email['text'].str.lower()
df_email.dropna(inplace = True)
for index, row in df_email.iterrows():
    token_word = nltk.tokenize.word_tokenize(row[0])
    #print(token_word)
    filtered_token = []
    for word in token_word:
        if word not in stop_words:
            filtered_token.append(word)


    filtered_sentence = " ".join(filtered_token)

    df_email.loc[index, 'text'] = filtered_sentence

df_email['text'] = df_email['text'].apply(stem_text)



##For Testing
# df_txt = df_email['text'].str.replace(r'\|', ' ').str.cat(sep=' ')
# words = nltk.tokenize.word_tokenize(df_txt)
# word_dist = nltk.FreqDist(words)
# df_freq = pd.DataFrame(word_dist.most_common(50),
#                     columns=['Word', 'Frequency'])

# print(df_freq)

texts = []

for index, row in df_email.iterrows():
    texts.append(row[0])

cv = CountVectorizer(max_features = 40000)

X = cv.fit_transform(texts).toarray()
#len(X[0])
y = df_email.iloc[:,1].values

X_train, X_init, y_train, y_init = train_test_split(X, y, test_size = 0.30, random_state = 0)

X_test, X_valid, y_test, y_valid = train_test_split(X_init, y_init, test_size = 0.50, random_state = 0)

###############################################################################################
models = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Trees', 'Ensemble']
df_results = pd.DataFrame({'Model': models, 'Accuracy': models, 'Precision': models, 
                            'Sensitivity': models, 'Specificity':models})


df_results = df_results.set_index(['Model'])
print(df_results)

########################################################################################
NB = GaussianNB()

NB.fit(X_train, y_train)

y_pred = NB.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)


tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("Bayes")
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
#######################################################################################
LR = LogisticRegression()

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)


tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("LogReg")
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
########################################################################################
SVM = SVC(kernel = 'linear', random_state = 0)

SVM.fit(X_train, y_train)

y_pred = SVM.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)



tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("SVM")
print(conf_matrix)
ConfMatrixDisp(y_test, y_pred, SVM.classes_, 'SVM', 'SVM')
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

########################################################################################

DT = DecisionTreeClassifier()

DT.fit(X_train, y_train)

y_pred = DT.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("Decision Tree")
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

#y_pred_proba = DT.predict_proba(X_test)[::, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test,  y_pred)
auc_dt = roc_auc_score(y_test, y_pred)
plt.plot(fpr_dt,tpr_dt,label="AUC="+str(auc_dt))
plt.title('ROC Curve for Decision Tree')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_DT.png')
plt.clf()

# graph = tree.plot_tree(DT)
# plt.show()

################################################################################################

Ensamble = VotingClassifier([('LR', LR), ('NB', NB), ('DT', DT)], voting = 'hard')

Ensamble.fit(X_train, y_train)

y_pred = Ensamble.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn+fp)

print("Ensemble")
print(conf_matrix)
print(accuracy_score(y_test, y_pred))

df_results.loc['Ensemble', 'Accuracy'] = accuracy_score(y_test, y_pred)

print(precision_score(y_test, y_pred))
df_results.loc['Ensemble', 'Precision'] = precision_score(y_test, y_pred)

print(recall_score(y_test, y_pred))
df_results.loc['Ensemble', 'Sensitivity'] = recall_score(y_test, y_pred)

print(specificity)
df_results.loc['Ensemble', 'Specificity'] = specificity

#y_pred_proba = Ensamble.predict_proba(X_test)[::, 1]
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
#Ensemble is my chosen model

df_chosen = pd.DataFrame({'Model': ['Ensemble'], 'Accuracy': ['Ensemble'], 'Precision': ['Ensemble'], 
                            'Sensitivity': ['Ensemble'], 'Specificity':['Ensemble']})

df_chosen = df_chosen.set_index(['Model'])
y_pred = Ensamble.predict(X_valid)

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
fpr_ensch, tpr_ensch, _ = roc_curve(y_valid,  y_pred)
auc_ensch = roc_auc_score(y_valid, y_pred)
plt.plot(fpr_ensch,tpr_ensch,label="AUC="+str(auc_ensch))
plt.title('ROC Curve for Ensemble')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC_Ensemble_chosen.png')
plt.clf()

df_chosen.to_excel('EmailChosen.xlsx')

fig, ax = plt.subplots(figsize = (6,6))

#NB
plt.plot(fpr_nb, tpr_nb, label = "Naive Bayes AUC=" + str(auc_nb.round(4)), color = "purple", linewidth = 4, alpha = 0.5)
plt.plot(fpr_lr, tpr_lr, label = "Log. Regression AUC=" + str(auc_lr.round(4)), color = "red", linewidth = 4, alpha = 0.5)
plt.plot(fpr_svm, tpr_svm, label = "SVM AUC=" + str(auc_svm.round(4)), color = "blue", linewidth = 4, alpha = 0.5)
plt.plot(fpr_dt, tpr_dt, label = "Dec. Trees AUC=" + str(auc_dt.round(4)), color = "orange", linewidth = 4, alpha = 0.5)
plt.plot(fpr_ens, tpr_ens, label = "Ensemble AUC=" + str(auc_ens.round(4)), color = "green", linewidth = 4, alpha = 0.5)

# plt.plot(fpr_nb, tpr_nb, "b^", label = "Naive Bayes AUC=" + str(auc_nb.round(4)), linewidth = 4,)
# plt.plot(fpr_lr, tpr_lr, "rD", label = "Log. Regression AUC=" + str(auc_lr.round(4)), linewidth = 4,)
# plt.plot(fpr_svm, tpr_svm, "ms", label = "SVM AUC=" + str(auc_svm.round(4)), linewidth = 4,)
# plt.plot(fpr_dt, tpr_dt, "c,", label = "Dec. Trees AUC=" + str(auc_dt.round(4)), linewidth = 4,)
# plt.plot(fpr_ens, tpr_ens, "g+", label = "Ensemble AUC=" + str(auc_ens.round(4)), linewidth = 4,)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Email Models")
plt.legend()
plt.savefig("AllROCEmail.png")


df_results.to_excel('Initialresults.xlsx')
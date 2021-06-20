#Katherine Yu

#425 project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import plot_precision_recall_curve
from sklearn import svm
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report






########import data
pd.set_option('display.max_columns', None)
df = pd.read_csv('creditcard.csv')

#########data description
print(df.head())
print(df.info())
print(df.describe())

#########data histogram
df.hist(bins = 50)
plt.show()


#######data processing
#mms = MinMaxScaler()
#df[['Amount']] = mms.fit_transform(df[['Amount']]) #scale amount only since the others that have undergone PCA
                                                    # are already normalized

rbs = RobustScaler()
df[['Amount']] = rbs.fit_transform(df[['Amount']]) #scale amount only since the others that have undergone PCA
                                                    # are already normalized


df = df.drop_duplicates()
print(df)

#######data exploration
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='BrBG') #correlation heatmap
plt.show()

outlier_count = {} #outliers
for column in df:
    temp = df[column]
    temp = np.array(sorted(temp))
    Q1,Q3 = np.percentile(temp, [25, 75])
    print(Q1, Q3)
    IQR = Q3-Q1
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    sum = len(temp[(temp < lower_range) | (temp > upper_range)])
    outlier_count[column] = sum


print(outlier_count) #v28 and 29 have many outliers

X = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,29]]
print(X)#all other columns except time, V27, V28 and the target
y = df.iloc[:,-1] #target

#######splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(np.mean(y_train), np.mean(y_test)) #have the almost the same positive classes in each

########logistic regression model (with cross validation)
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
print(logreg.coef_, logreg.intercept_)

#using sm library to get significance - not much meaningful results.
#Possibly complete quasi-separation: A fraction 0.34 of observations can be
#perfectly predicted. This might indicate that there is complete
#quasi-separation. In this case some parameters will not be identified.
#logit_model=sm.Logit(y_train,X_train)
#result=logit_model.fit()
#print(result.summary())


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

scores = cross_val_score(logreg, X_train, y_train, cv=5)
print('Logistic regression cross validation:')
print(scores)

#######logistic regression - ROC and precision recall
print('sklearn ROC curve for logit:')
metrics.plot_roc_curve(logreg, X_test, y_test)
plt.show()

disp = plot_precision_recall_curve(logreg, X_test, y_test)
plt.show()



#######logistic regression - undersampling - model (with cross validation)
df = df.sample(frac=1, random_state=42) #shuffle dataset

df_fraud = df.loc[df['Class'] == 1]
df_good = df.loc[df['Class'] == 0][:473]

df_undersampled = pd.concat([df_fraud, df_good])
df_undersampled = df_undersampled.sample(frac=1, random_state=42) #shuffle again for preparation for test train split

#print(df_fraud)

X_u = df_undersampled.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,29]] #all other columns except time, V28, V29 and the target
y_u = df_undersampled.iloc[:,-1]

heatmap = sns.heatmap(df_undersampled.corr(), vmin=-1, vmax=1, cmap='BrBG') #correlation heatmap
plt.show()

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_u, y_u, test_size=0.5, random_state=42)

print(np.mean(y_train_u), np.mean(y_test_u)) #have the almost the same positive classes in each

#######logistic regression model (with cross validation)
logreg_u = LogisticRegression(max_iter=500)
logreg_u.fit(X_train_u, y_train_u)
print(logreg_u.coef_, logreg_u.intercept_)

########using sm library to get significance
logit_model=sm.Logit(y_train_u,X_train_u)
result=logit_model.fit()
print(result.summary())
#Possibly complete quasi-separation: A fraction 0.39 of observations can be
#perfectly predicted. This might indicate that there is complete
#quasi-separation. In this case some parameters will not be identified.

y_pred_u = logreg_u.predict(X_test_u)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_u.score(X_test_u, y_test_u)))

scores_u = cross_val_score(logreg_u, X_train_u, y_train_u, cv=3)
print('Logistic regression cross validation:')
print(scores_u)

########logistic regression - ROC and precision recall - undersampled
sklearn_pred_proba_u = logreg_u.predict_proba(X_test_u)
print('sklearn ROC curve for logit - undersampled:')
metrics.plot_roc_curve(logreg_u, X_test_u, y_test_u)
plt.show()

disp_u = plot_precision_recall_curve(logreg_u, X_test_u, y_test_u)
plt.show()

#########SVM - grid search CV on undersampled data
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_params_svm = svc_param_selection(X_train_u, y_train_u, 3)
print(best_params_svm)

svm_best = svm.SVC(kernel='rbf', C = 10, gamma = .01)
svm_best.fit(X_train_u, y_train_u)
svm_pred = svm_best.predict(X_test_u)


print('Accuracy of logistic svm classifier on test set: {:.2f}'.format(svm_best.score(X_test_u, y_test_u)))

########SVM - confusion matrix
print(confusion_matrix(y_test_u, svm_pred, labels=[0,1]))
print(precision_recall_fscore_support(y_test_u, svm_pred))
print(classification_report(y_test_u, svm_pred, labels=[0, 1]))

#########logistic regression undersampled confusion matrix
print(confusion_matrix(y_test_u, y_pred_u, labels=[0,1]))
print(classification_report(y_test_u, y_pred_u, labels=[0, 1]))


#########logistic regression confusion matrix with adjusted threshold
print(sklearn_pred_proba_u)
decisions = (sklearn_pred_proba_u[:,1] >= 0.4).astype(int)
print(decisions)

print(confusion_matrix(y_test_u, decisions, labels=[0,1]))
print(classification_report(y_test_u, decisions, labels=[0, 1]))

decisions_3 = (sklearn_pred_proba_u[:,1] >= 0.3).astype(int)

print(confusion_matrix(y_test_u, decisions_3, labels=[0,1]))
print(classification_report(y_test_u, decisions_3, labels=[0, 1]))




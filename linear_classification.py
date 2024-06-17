import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def prepare_data(df, train_size=None,shuffle=True, random_state=None):

    df = df.drop(['Month', 'Browser', 'OperatingSystems'], axis=1)

    df['Revenue'] = df['Revenue'].astype(int)

    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])

    X = df.drop('Revenue', axis=1)

    y = df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)

    return X_train, X_test, y_train, y_test


df = pd.read_csv('project2_dataset.csv')

    
X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, shuffle=True, random_state=42)

scaler = MinMaxScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

lr = LogisticRegression(penalty=None, max_iter=30000)
lr.fit(X_train_scaler, y_train)

y_hat_train = lr.predict(X_train_scaler)
y_hat_test = lr.predict(X_test_scaler)

acc_train = metrics.accuracy_score(y_train, y_hat_train)
acc_test = metrics.accuracy_score(y_test, y_hat_test)

print('Accuracy training = {}'.format(acc_train))
print('Accuracy test = {}'.format(acc_test))

c_m = metrics.confusion_matrix(y_test, y_hat_test)
print(c_m)

disp = metrics.ConfusionMatrixDisplay(c_m)
disp.plot()
plt.show()





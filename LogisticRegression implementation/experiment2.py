import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegressionEP34


data = load_breast_cancer()
x, y = data.data, data.target


accuracies = []

start_time = time.time()


for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    
    model = LogisticRegressionEP34(lr=0.01)
    model.fit(x_train, y_train,batch_size=64)
    
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for iteration {i}: {accuracy:.4f}")


mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)


end_time = time.time()
run_time = end_time - start_time

# results
print("-----------------------------")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Run Time: {run_time:.2f} seconds") 
print("-----------------------------")
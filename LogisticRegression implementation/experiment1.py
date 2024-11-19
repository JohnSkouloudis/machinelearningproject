from generate_dataset import generate_binary_problem
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegressionEP34


#Test 1
print("-----------------Test 1--------------------")

c1 = np.array([[0, 8], [0, 8]])

x1,y1 = generate_binary_problem(c1,1000)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y1,train_size=0.7)

print(f"x_train={x_train1.shape}, x_test={x_test1.shape}, y_train={y_train1.shape}, y_test={y_test1.shape}")

model1 = LogisticRegressionEP34(lr=1e-2)
model1.fit(x_train1, y_train1, show_line=True)

pred1 = model1.predict(x_test1)

acc1 = accuracy_score(y_test1, pred1)


print(f"Accuracy for {c1}={acc1}")
print("-------------------------------------")


#Test 2

print("-----------------Test 2--------------------")
c2 = np.array([[0, 1], [0, 3]])

x2,y2 = generate_binary_problem(c2,1000)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2,y2,train_size=0.7)

print(f"x_train={x_train2.shape}, x_test={x_test2.shape}, y_train={y_train2.shape}, y_test={y_test2.shape}")

model2= LogisticRegressionEP34(lr=1e-2)
model2.fit(x_train2, y_train2, show_line=True)

pred2 = model1.predict(x_test2)

acc2 = accuracy_score(y_test2, pred2)


print(f"Accuracy for {c2}={acc2}")
print("-------------------------------------======")


#Test 3
print("-----------------Test 3--------------------")

c3 = np.array([[0, 1], [0, 1]])

x3,y3 = generate_binary_problem(c3,1000)

x_train3, x_test3, y_train3, y_test3 = train_test_split(x3,y3,train_size=0.7)

print(f"x_train={x_train3.shape}, x_test={x_test3.shape}, y_train={y_train3.shape}, y_test={y_test3.shape}")

model3= LogisticRegressionEP34(lr=1e-2)
model3.fit(x_train3, y_train3, show_line=True)

pred3 = model3.predict(x_test3)

acc3 = accuracy_score(y_test3, pred3)

print(f"Accuracy for {c3}={acc3}")
print("-------------------------------------------")




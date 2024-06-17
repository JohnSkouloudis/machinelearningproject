import numpy as np 
import pandas as pd 



data = pd.read_csv('project2_dataset.csv')

records = len(data)

print(f'1.The number of records is: {records}')

num_revenues = (data['Revenue'] == True).sum()

print(f'The number of revenues is: {num_revenues}')


print(f'2.The percentage of users that bought is: {( ( num_revenues/records ) *100 ):.2f}%')

print(f'Accuracy of a model that always predicts that a user will not buy: {( (data['Revenue'] == False).mean() * 100 ):.2f}%')



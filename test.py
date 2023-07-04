from river import datasets
from river import stream
import pandas as pd
import csv

dataset = datasets.Phishing()
dataset

for x, y in dataset:
    pass

x, y = next(iter(dataset))
print(x)
print(type(x))
print(y)
print(type(y))

# X extract
df_x = pd.read_csv('artificial/sea/SEA_training_data.csv', header=None)
x = df_x.to_dict('records')

# y extract
df = pd.read_csv('artificial/sea/SEA_training_class.csv', header=None)
y = df[0].tolist()

for idx, x in enumerate(x):
    print(x)
    print(type(x))
    print(y[idx])
    print(type(y[idx]))
    break


#df_x = pd.read_csv('artificial/sea/SEA_training_data.csv', header=None)
#for idx, x in enumerate(dataset):
#    print(x, y[idx])
#x = df_x.to_dict('records')
#print(x)



#df = pd.read_csv('artificial/sea/SEA_training_class.csv', header=None)
#print(df)
#print(df.columns)
#y = df[0].tolist()
#print(y[0])
#print(y[1])



#dataset = stream.iter_csv('artificial/sea/SEA_training_data.csv')
#for idx, x in enumerate(dataset):
#    print(x, y[idx])

    


from river import datasets
from river import stream
from river import tree
from river import neighbors
from river import preprocessing
from river import metrics   
import pandas as pd
import csv


# X extract
df_x = pd.read_csv('artificial/sea/SEA_training_data.csv', header=None)
x = df_x.to_dict('records')

# y extract
df = pd.read_csv('artificial/sea/SEA_training_class.csv', header=None)
y = df[0].tolist()

# Tree Classifier
modelEFDT = tree.ExtremelyFastDecisionTreeClassifier()
modelKNN = (
    preprocessing.StandardScaler() |
    neighbors.KNNClassifier(window_size=50)
)

# Metrics
metric_acuracia = metrics.Accuracy()
metric_m_confusao = metrics.ConfusionMatrix()
metric_precisao = metrics.Precision()
metric_recall = metrics.Recall()
metric_f1 = metrics.F1()
metric_report = metrics.ClassificationReport()

# Evaluate
model = modelEFDT
for idx, x in enumerate(x):
    print(x, y[idx])

    # Test Than Train
    # Test
    y_pred = model.predict_one(x)
    # Train
    model.learn_one(x, y[idx])

    if y_pred is not None:
        metric_acuracia.update(y[idx], y_pred)
        metric_m_confusao.update(y[idx], y_pred)
        metric_precisao.update(y[idx], y_pred)
        metric_recall.update(y[idx], y_pred)
        metric_f1.update(y[idx], y_pred)
        metric_report.update(y[idx], y_pred)

    
print("\nMatriz de Confusão:\n",metric_m_confusao)
print("\n")
print(metric_acuracia)
print(metric_precisao)
print(metric_recall)
print(metric_f1)
print("\n\n")
print(metric_report)

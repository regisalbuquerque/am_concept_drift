from evaluate import ttt
from river import tree
from river import metrics
from river import stream

modelEFDT = tree.ExtremelyFastDecisionTreeClassifier()
metric_acuracia = metrics.Accuracy()

params = {
    'converters': {'x': float, 'y': float, 'z': float},
    'target': 'labels'
}

dataset = stream.iter_csv("artificial/sea/sea.csv", **params)


ttt.TestThenTrain.evaluate(modelEFDT, dataset, metric_acuracia)
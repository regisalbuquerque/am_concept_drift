from evaluate import ttt

from river import ensemble, forest, linear_model, neighbors, preprocessing, tree


algos = {
    'ensemble': ensemble.LeveragingBaggingClassifier(tree.HoeffdingTreeClassifier()),
    'tree': tree.ExtremelyFastDecisionTreeClassifier(),
    'forest': forest.ARFClassifier(),
    'neighbors': neighbors.SAMKNNClassifier()
}

bases = {
    # 'chess': {
    #     'data': 'datasets/artificial/chess/transientChessboard.data',
    #     'labels': 'datasets/artificial/chess/transientChessboard.labels',
    #     'fieldnames': {'x', 'y'},
    # },
    # 'rbf': {
    #     'data': 'datasets/artificial/rbf/movingRBF.data',
    #     'labels': 'datasets/artificial/rbf/movingRBF.labels',
    #     'fieldnames': {'x0', 'x1', 'x2', 'x3', 'x4',
    #                    'x5', 'x6', 'x7', 'x8', 'x9'},
    # },
    # 'sea': {
    #     'data': 'datasets/artificial/sea/SEA_training_data_xy.csv',
    #     'labels': 'datasets/artificial/sea/SEA_training_class.csv',
    #     'fieldnames': {'x', 'y'},
    # },
    # 'squares': {
    #     'data': 'datasets/artificial/movingSquares/movingSquares.data',
    #     'labels': 'datasets/artificial/movingSquares/movingSquares.labels',
    #     'fieldnames': {'x', 'y'},
    # },
    'poker': {
        'data': 'datasets/realWord/poker/poker.data',
        'labels': 'datasets/realWord/poker/poker.labels',
        'fieldnames': {'x0', 'x1', 'x2', 'x3', 'x4',
                       'x5', 'x6', 'x7', 'x8', 'x9'},
    },
    # 'weather': {
    #     'data': 'datasets/realWord/weather/NEweather_data.csv',
    #     'labels': 'datasets/realWord/weather/NEweather_class.csv',
    #     'fieldnames': {'x0', 'x1', 'x2', 'x3', 
    #                    'x4', 'x5', 'x6', 'x7'},
    # },
}

all_metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1',
    'GeometricMean',
]


ttt.TestThenTrain.evaluate_all(algos, bases, all_metrics)
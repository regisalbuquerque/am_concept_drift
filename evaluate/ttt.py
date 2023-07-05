class TestThenTrain:

    def evaluate(model, dataset, metric):
        for x, y in dataset:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            if y_pred is not None:
                metric.update(y, y_pred)
        print(metric)

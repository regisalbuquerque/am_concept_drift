import csv
import copy
from datetime import datetime
from river import metrics

class TestThenTrain:

    def evaluate(model, dataset, metric):
        for x, y in dataset:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            if y_pred is not None:
                metric.update(y, y_pred)
        print(metric)

    def evaluate_all(algos, bases, all_metrics):
        
        for bas in bases:

            print(f'starting {bas}, at {datetime.now()}')

            try:
                f = open(bases[bas]['labels'], 'r')
                y = []
                for line in f.readlines():
                    y.append(int(line))
                f.close()
                print(f'import {bas} labels, ok!')
            except:
                print('Y not found')
            try:
                f = open(bases[bas]['data'], 'r')
                x = csv.DictReader(
                    f,
                    delimiter=' ',
                    quoting=csv.QUOTE_NONNUMERIC,
                    fieldnames=bases[bas]['fieldnames']
                )
                print(f'import {bas} data, ok!')
            except:
                print('X not found')


            result = {}
            for mod in algos:
                _metric = {'y_pred': None}
                for m in all_metrics:
                    _m = getattr(metrics, m)()
                    _metric.update({m: _m})
                result.update({mod: _metric})

            try:
                w = open(f'results/{bas}.csv', 'w')
                writer = csv.writer(w)
                writer.writerow(['instance',
                                'model', 
                                'Accuracy',
                                'Precision',
                                'Recall',
                                'F1',
                                'GeometricMean'
                                ])
            except:
                print('Result file can\'t be opened')

            # Init Classifiers
            models = {}
            for mod in algos:
                models[mod] = copy.deepcopy(algos[mod])

            for i, _x in enumerate(x):
                for mod in algos:

                    # Test Then Train
                    # Test
                    result[mod]['y_pred'] = models[mod].predict_one(_x)
                    # Train
                    models[mod].learn_one(_x, y[i])

                    if result[mod]['y_pred'] is not None:
                        a = [i, mod]
                        for m in all_metrics:
                            result[mod][m].update(y[i], result[mod]['y_pred'])
                            a.append(result[mod][m].get())

                        writer.writerow(a)
            print(f'finishing {bas}, at {datetime.now()}')
            w.close()
            f.close()

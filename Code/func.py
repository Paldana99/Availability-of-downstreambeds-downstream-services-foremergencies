import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def get_n_visit(data):
    data = data.sort_values(['patientid', 'case_id'])
    n = len(data)

    n_visit = [0]
    for i in range(1, n):
        pid = data.patientid.iloc[i]
        if data.patientid.iloc[i-1] == pid:
            n_visit.append(n_visit[len(n_visit)-1] + 1)
        else:
            n_visit.append(0)

    data['n_visit'] = n_visit
    data = data.sort_index()
    return data

def get_day_visit(data):
    n = len(data)

    n_visit = list()

    for i in range(0, n):
        stay = data.Stay.iloc[i]
        # print(stay)
        if stay == 'More than 100 Days':
            day = randint(100, 200)
        else:
            stay = stay.split('-')
            day = randint(int(stay[0]), int(stay[1]))
        n_visit.append(day)
    data['n_day_stay'] = n_visit
    return data

def print_roc(y_test, y_prob):
    for i in range(0, 1):
        y_prob = y_prob[:,0]
        y_exp = ((y_test.to_numpy()) == 0)
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_exp, y_prob)
        print('roc_auc_score for DecisionTree: ', roc_auc_score(y_exp, y_prob))
        plt.subplots(1, figsize=(10,10))
        plt.title('Receiver Operating Characteristic - DecisionTree')
        plt.plot(false_positive_rate1, true_positive_rate1)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return
        

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def calibrate_probability(cat_model, X_to_pred, X_train, y_train, y_pred, num_bins=1000):
    def transform(x):
        return np.log(x/(1-x))
    def give_ece_data(preds,bins,y_valid):
        sorted_ind = np.argsort(preds)
        predicted_bins = [[] for _ in range(bins)]
        actual_counters = [[] for _ in range(bins)]
        counters = [[] for _ in range(bins)]
        index = 0
        length_array = len(sorted_ind)
        step = 1.*length_array//bins
        for _ in range(bins):
            current = int(step*index)
            next_ = int(step*(index+1))
            predicted_bins[index] = np.mean(preds[sorted_ind[current:next_]])
            actual_counters[index] = np.mean(y_valid[sorted_ind[current:next_]])
            counters[index] = len(y_valid[sorted_ind[current:next_]])
            index += 1
        return predicted_bins,actual_counters,counters
        
    print('BEFORE CALIBRATION')
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, y_pred, n_bins=100)
    plt.ioff()

    fig, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-')
    plt.plot([0, 1], [0, 1], '--', color='gray')

    sns.despine(left=True, bottom=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.title("Предсказания до калибровки", fontsize=20);
    plt.show()
    print(fig)

    tree_model = DecisionTreeClassifier(min_samples_leaf=1700,max_depth=5)
    tree_model.fit(train.drop(columns=['default6', 'contract_id']),train['default6'])

    TREE = tree_model.tree_
    indexes = TREE.apply(X_train.values.astype(np.float32))
    # predicts_from_xgboost = model_xgb.predict(x_valid)
    predicts_from_xgboost = y_pred.reshape((-1,1))

    
    log_reg_dict = {}
    nodes = np.unique(indexes)
    model = sklearn.linear_model.LogisticRegression()
    for node in tqdm(nodes):
        model.fit(transform(predicts_from_xgboost[indexes==node]),y_train[indexes==node])
        log_reg_dict[node] = model
    
    indexes_test = TREE.apply(X_train.values.astype(np.float32))
    predicts_from_xgboost_test = y_pred.reshape((-1,1))
    predicts_calibrated = np.zeros_like(predicts_from_xgboost_test)

    
    for node in tqdm(log_reg_dict.keys()):
        predicts_calibrated[indexes_test==node] = log_reg_dict[node].\
            predict_proba(transform(predicts_from_xgboost_test[indexes_test==node]))[:,1].reshape((-1,1))
        
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, predicts_calibrated, n_bins=100)
    plt.ioff()

    fig, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-')
    plt.plot([0, 1], [0, 1], '--', color='gray')

    sns.despine(left=True, bottom=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.title("Предсказания после калибровки", fontsize=20);
    plt.show()
    print(fig)

    indexes_test = TREE.apply(X_to_pred.values.astype(np.float32))
    predicts_from_xgboost_test = cat_model.predict_proba(X_to_pred)[:, 1]
    predicts_from_xgboost_test = predicts_from_xgboost_test.reshape((-1,1))
    predicts_calibrated = np.zeros_like(predicts_from_xgboost_test)


    for node in tqdm(log_reg_dict.keys()):
        predicts_calibrated[indexes_test==node] = log_reg_dict[node].\
        predict_proba(transform(predicts_from_xgboost_test[indexes_test==node]))[:,1].reshape((-1,1))

    return predicts_calibrated
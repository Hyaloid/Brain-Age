import pickle
import pandas as pd


def save_model(file_name, models):
    with open(file_name, 'wb') as f:
        pickle.dump(models, f)
    print('model saved.!')


def write2csv(test_id, pred, filename=''):
    sub = pd.DataFrame()
    sub['subject_ID'] = test_id
    sub['年龄'] = pred
    sub.to_csv(filename, index=False)

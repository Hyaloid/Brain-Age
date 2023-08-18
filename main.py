from preprocessing.data_preprocessing import data_preprocessing
from dataset.load_data import train, test
from modeling.model import my_stacking_model, my_averaged_model
from utils.score import model_train, model_pred
from utils.save_file import write2csv
import pandas as pd


if __name__ == '__main__':
    # preprocessing
    train, y_train, test, test_id = data_preprocessing(train, test)

    # averaged_model
    # averaged_models = my_averaged_model()
    # model_pred(test, file_name='GBoost_KRR_LGB_lasso.pkl')

    # stacking model
    stacked_averaged_model = my_stacking_model()
    model_train(stacked_averaged_model, train, y_train, model_name='stacked_averaged_model1')   # K Fold Validation
    stacked_averaged_model.fit(train.values, y_train)
    # pred = stacked_averaged_model.predict(test.values)
    pred = model_pred(test, 'stacked_averaged_model1.pkl')
    write2csv(test_id, pred, filename='submission.csv')

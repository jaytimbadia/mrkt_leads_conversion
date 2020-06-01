import pandas as pd
import pickle
import json
import argparse


def predict(model_name, csv_file):
    # Loads and predict the model.
    # input required: model name (lr || svm) (optional), csv_file_predict (optional)
    try:
        data = pd.read_csv(csv_file, usecols=['landing_page_id', 'origin'])

        model_data_dir = 'model/model_data/'
        with open(model_data_dir + 'landing_page.json', 'r') as fl:
            landing_page =  json.load(fl)
        with open(model_data_dir + 'origin.json', 'r') as fo:
            origin_json = json.load(fo)


        cleanup = {"landing_page_id": landing_page, "origin": origin_json}
        data.replace(cleanup, inplace = True)
        data.astype({'landing_page_id': 'int32'}).dtypes
        data.astype({'origin': 'int32'}).dtypes

        # load the model
        if model_name == 'lr':
            filename = 'logistic.pkl'
        elif model_name == 'svm':
            filename = 'svm.pkl'
        else:
            raise Exception('Sorry, currently do not privide support for {} model'.format(model_name))

        training_dir = 'model/trained_models/'
        with open(training_dir + filename, 'rb') as file:
            clf = pickle.load(file)

        res = clf.predict_proba(data.iloc[:,0:2].values)
        print('Your Results: ')
        print(res)

    except Exception as error:
        raise Exception(error)


# predict(model_name, inputfile)
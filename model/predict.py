import pandas as pd
import pickle
import json
import argparse


arg_parse = argparse.ArgumentParser()

arg_parse.add_argument('-m', '--modelname', required=False, help='choose from lr or svm', default='svm')
arg_parse.add_argument('-f', '--inputfile', required=False, help='path to data file (.csv)', default='model_data/default_test.csv')

args = vars(arg_parse.parse_args())


model_name = args['modelname']
inputfile = args['inputfile']
args = None


def predict(model_name, csv_file):
    # Loads and predict the model.
    # input required: model name (lr || svm) (optional), csv_file_predict (optional)
    try:
        data = pd.read_csv(csv_file, usecols=['landing_page_id', 'origin'])

        model_data_dir = 'model_data/'
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

        training_dir = 'trained_models/'
        with open(training_dir + filename, 'rb') as file:
            clf = pickle.load(file)

        res = clf.predict_proba(data.iloc[:,0:2].values)
        print('Your Results: ')
        print(res)

    except Exception as error:
        raise Exception(error)


predict(model_name, inputfile)
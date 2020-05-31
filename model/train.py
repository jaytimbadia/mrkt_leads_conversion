from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse
from sklearn import svm
import pandas as pd
import functools
import time
import pickle
import json

arg_parse = argparse.ArgumentParser()

arg_parse.add_argument('-m', '--modelname', required=False, help='choose from lr or svm', default='svm')
arg_parse.add_argument('-f', '--inputfile', required=True, help='path to data file (.csv)')

args = vars(arg_parse.parse_args())

model_name = args['modelname']
inputfile = args['inputfile']
args = None


def timer(func):
    # timer wrap to just monitor the time taken by a pirticular function to execute.
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


@timer
def model_preprocessing(data):
    # preprocessnig the incoming data for pirtcular landing_page_id and origin.
    # Also splits data to train  & evaluation.
    try:
        landing_page_json = dict()
        origin_json = dict()
        model_data_dir = 'model_data/'
        _lcounter, _ocounter = 0, 0
        for index, row in data.iterrows():
            if row['landing_page_id'] not in landing_page_json.keys():
                _lcounter += 1
                landing_page_json[row['landing_page_id']] = _lcounter
            if row['origin'] not in origin_json.keys():
                _ocounter += 1
                origin_json[row['origin']] = _ocounter

        cleanup = {"landing_page_id": landing_page_json, "origin": origin_json}
        data.replace(cleanup, inplace = True)

        with open(model_data_dir + 'landing_page.json', 'w') as fl:
            json.dump(landing_page_json, fl, indent=4)
        with open(model_data_dir + 'origin.json', 'w') as fo:
            json.dump(origin_json, fo, indent=4)

        print('Data processed successfully to features.')
        print('Preparing data for training & evaluation.')

        train, evals = train_test_split(data, test_size=0.15)
        train.to_csv('model_data/train.csv')
        evals.to_csv('model_data/eval.csv')
        print('Training data examples: {}'.format(len(train)))
        print('Evaluation data examples: {}'.format(len(evals)))
    except Exception as error:
        print(f'Error Occured: {str(error)}')


@timer
def model_train(csv_file, model_name):
    # Trains & save the model baed on inputs passed.
    # input required: model name (lr || svm) (optional), csv_file_train_with_labels
    try:

        data = pd.read_csv(csv_file, usecols=['landing_page_id', 'origin', 'label'])
        model_preprocessing(data)

        if model_name == 'lr':
            filename = 'logistic.pkl'
            clf = LogisticRegression(C=1.0)  # Note that C controls the effect regularization
        elif model_name == 'svm':
            filename = 'svm.pkl'
            clf = svm.SVC(C=1.0, probability=True)
        else:
            raise Exception('Sorry, currently do not privide support for {} model'.format(model_name))

        training_data = pd.read_csv('model_data/train.csv', dtype={'label': int})

        X = training_data.iloc[:,1:3].values
        clf.fit(X=X, y=training_data.label.values)

        trained_dir = 'trained_models/'

        with open(trained_dir + filename, 'wb') as file:
            pickle.dump(clf, file)

        print('Model trained successfully and saved.')
    except Exception as error:
        print('Error Occured: {}'.format(str(error)))
        raise Exception(error)

# Train Coomand: python train.py -m "lr" -f "../data/test_merge.csv"
# Note: test_merge.csv is the entire data and later splitted to train & eval
model_train(inputfile, model_name)



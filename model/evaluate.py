import pandas as pd
import pickle
import json


def evalaute(model_name, csv_file):
    # Loads and evaluate the model.
    # input required: model name (lr || svm) (optional)

    try:
        data = pd.read_csv(csv_file, usecols=['landing_page_id', 'origin', 'label'])

        true_labels = data['label'].values.tolist()

        data.astype({'landing_page_id': 'int32'}).dtypes
        data.astype({'origin': 'int32'}).dtypes
        data.astype({'label': 'int32'}).dtypes

        # load the model
        if model_name == 'lr':
            print('Evaluating on logistic model.')
            filename = 'logistic.pkl'
        elif model_name == 'svm':
            print('Evaluating on svm model.')
            filename = 'svm.pkl'
        else:
            raise Exception('Sorry, currently do not privide support for {} model'.format(model_name))

        training_dir = 'model/trained_models/'
        with open(training_dir + filename, 'rb') as file:
            clf = pickle.load(file)

        res_prob = clf.score(data.iloc[:,0:2].values, true_labels)

        print('Accuracy on evaluation: ', res_prob)

    except Exception as error:
        raise Exception(error)

# Accuracy on evaluation:  0.7108433734939759
# evalaute(model_name, inputfile)
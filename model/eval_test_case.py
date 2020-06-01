import pandas as pd
import pickle
import json
import os

def version_shift():
    # Monitors the models as developer updates and checks for decision on making a version shift.

    try:

        data = pd.read_csv('model/model_data/eval.csv', usecols=['landing_page_id', 'origin', 'label'])
        true_labels = data['label'].values.tolist()
        data.astype({'landing_page_id': 'int32'}).dtypes
        data.astype({'origin': 'int32'}).dtypes
        data.astype({'label': 'int32'}).dtypes

        validate_dir = 'model/monitor/'
        training_dir = 'model/trained_models/'

        if os.path.exists(validate_dir + 'temp.txt'):
            os.remove(validate_dir + 'temp.txt')

        for model_name in ['lr', 'svm']:

            # load the model
            if model_name == 'lr':
                print('Evaluating on logistic model.')
                filename = 'logistic.pkl'
            elif model_name == 'svm':
                print('Evaluating on svm model.')
                filename = 'svm.pkl'
            with open(training_dir + filename, 'rb') as file:
                clf = pickle.load(file)

            res_prob = clf.score(data.iloc[:,0:2].values, true_labels)

            file = open(validate_dir + 'temp.txt', 'a')
            file.write(model_name + ':' + str(res_prob) + '\n')
            file.close()

        file_read = open(validate_dir + 'rollback.txt', 'r')
        file_check = open(validate_dir + 'temp.txt', 'r')
        lines1 = file_read.readlines()
        lines2 = file_check.readlines()

        for i, j in zip(lines1, lines2):
            acc1 = float(i.split(':')[1])
            acc2 = float(j.split(':')[1])
            if acc1 > acc2:
                print('***********'+'Your model does not shows considerable improvements. Version shift not possible.')
                return False
        print('***********' + 'Your model shows considerable improvements. You are good for version shift.')
        return True

    except Exception as error:
        raise Exception(error)





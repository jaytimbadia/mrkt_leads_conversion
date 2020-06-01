import argparse
from model.predict import predict
from model.train import model_train
from model.evaluate import evalaute


arg_parse = argparse.ArgumentParser()

arg_parse.add_argument('-w', '--script_name', required=True, help='which script to run.[Predict, Train, Evaluate]')
arg_parse.add_argument('-m', '--modelname', required=False, help='choose from lr or svm', default='svm')
arg_parse.add_argument('-f', '--inputfile', required=False, help='path to data file (.csv)', default='model/model_data/default_test.csv')

args = vars(arg_parse.parse_args())

script_name = args['script_name']
model_name = args['modelname']
inputfile = args['inputfile']
args = None


if script_name == 'Predict':
    predict(model_name, inputfile)
elif script_name == 'Train':
    if inputfile == None:
        raise Exception('Input csv trained data file is missing.')
    model_train(model_name, inputfile)
elif script_name == 'Evaluate':
    inputfile = 'model/model_data/eval.csv'
    evalaute(model_name, inputfile)
elif script_name == None:
    # It does automatic btw
    print('Please provide a valid method to run from [Predict, Train, Evaluate]')
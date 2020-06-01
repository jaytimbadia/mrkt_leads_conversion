# mrkt_leads_conversion
Using Machine learning to predict leads for conversion.  


# mrkt_leads_conversion
Using Machine learning to predict leads for conversion.  

This repository caters better marketing leads conversion for OSlist store using machine learning approach.

Repository walkthrough.

Dataset: https://www.kaggle.com/olistbr/marketing-funnel-olist
Data - This folder contains all the preprocssing required to convert the data kaggle downloaded format to machine learning model
intaking format.
This folder also contains sata visializtion code which further gets eloborated into the word document in the repository.

Model - This folder contains all the machine learning processing required to achieve the problem statement.
It has train, predict and evaluation scripts.

### Running Scripts
To run script: use run_scripts.py

usage: run_scipts.py [-h] -w SCRIPT_NAME [-m MODELNAME] [-f INPUTFILE]

optional arguments:
  -h, --help            show this help message and exit
  -w SCRIPT_NAME, --script_name SCRIPT_NAME
                        which script to run.[Predict, Train, Evaluate]
  -m MODELNAME, --modelname MODELNAME
                        choose from lr or svm
  -f INPUTFILE, --inputfile INPUTFILE
                        path to data file (.csv)
                        
For Predict, there is a default_csv placed, but can add your own too.
For Evaluation you do not need to provide file, as it will generate automatically from the training data you provide.
                        
### Jenkins Integration                        
Also, for Jenkins integration, we have made one version shifter file.
Whenever someone pushes code to release or wants to update the code to master,
they need to validate their build using version shift.bat.

This file contains a very simple test case validation.
If developer creates a new model, and if the performance exceeds the previous version's preformance,
build gets passed else codebase stays as it is.

This obviously can be extended further, but this is just one naive way to bring Jenkins into action and demonstrate complete
ML Devops Pipeline.

#### Note: To run version_shift.bat, kindly configure python environment and repo path as per your system python & download configuration.

### Docker
Kindly read dockers.txt for information.

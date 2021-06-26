# Disaster Response Pipeline Project

### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="instructions"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl <usegridsearch>`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://WORKSPACEID-3001.WORKSPACEDOMAIN based on your environment's path.

    - To find values for WORKSPACEID & WORKSPACEDOMAIN
        `env|grep WORK`


## Project Motivation<a name="motivation"></a>

For this project, I aimed to build a Machine Learning pipeline to analyze tweets during a natural disaster. I parse, clean, and tokenize the tweets, then I build a pipeline consisting of CountVectorizer, TfidTransformer, and RandomForestClassifier. Finally, I train and test the model. I output the model as a pickle file for use in a web app to display the results.

NOTE: This project also called for the use of GridSearchCV; however, the model training would never complete in my test environment. Therefore, I included code to employ GridSearchCV but it will only run if the user includes a third command line argument of "usegridsearch." If that argument is omitted, the project will run without that feature. Please note I could not test my implementation of GridSearchCV because it would never run to completion.

## File Descriptions <a name="files"></a>

data/disaster_messages.csv - tweets made during a natural disaster
data/disaster_categories.csv - categories of tweets
data/DisasterResponse.db - database containing results of nltk processing
data/process_data.py - python script to process tweets and generate database
models/train_classifier.py - python script to create, train, test, and store ML model based on nltk database
models/classifier.pkl - trained ML model
app/run.py - python script to run web app that displays ML model performance
results/results.txt - accuracy, 

## Results<a name="results"></a>

The main findings of the code can be explored using the web app. However, as described earlier, I could not make GridSearchCV complete its training in my test environment due to resource limitatoins. Therefore, the quality of the model is limited.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit goes to FigureEight for the data.  Otherwise, there are no licensing restrictions on this code.

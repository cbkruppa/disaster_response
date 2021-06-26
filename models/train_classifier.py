# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    """
    Loads data from database.
    
    Arguments:
    database_filepath: path to database containing data
    
    Returns:
    X: content of tweets
    y: categories of tweets as 0s and 1s
    cols: names of categories of tweets
    """
    # load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("MessagesCategories",engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    cols = df.columns[4:].tolist()
    return X, y, cols


def tokenize(text):
    """
    Cleans text of natural language tweets.
    
    Arguments:
    text: string containing a single tweet
    
    Returns:
    clean_tokens: list of cleaned tokens from input text
    """
    # tokens = word_tokenize(text)
    # Normalize text to exclude punctuation and tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    
    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through tokens removing stopwords and lemmatizing
    clean_tokens = []
    for tok in tokens:
        # Reduce words to their root form & lemmatize verbs by specifying pos
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok).lower().strip(), pos='v')
        # Remove stop words
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds machine learning model.
    
    Arguments:
    none
    
    Returns:
    pipeline: pipeline of machine learning model
    """
    # build pipeline consisting of vectorizer, transformer, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline

def build_CV_model():
    """
    Builds machine learning model.
    
    Arguments:
    none
    
    Returns:
    GridSearchCV pipeline of machine learning model
    """
    # build pipeline consisting of vectorizer, transformer, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # set parameters for GridSearch to test in its opitmization
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [5, 10]
    }
    
    # return pipeline
    return GridSearchCV(pipeline, param_grid=parameters, n_jobs = 1)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts target data using trained Pipeline model. Prints a classification report.
    
    Arguments:
    model: trained Pipeline model
    X_test: test data set to use for prediction
    Y_test: target data set to use for comparison
    
    Returns:
    none
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print classification report for all categories tested
    for i in range(len(category_names)):
        print("Classification report for " + category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
        print("Accuracy = " + str((Y_pred[:,i] == Y_test[:,i]).mean()))

def evaluate_CV_model(model, X_test, Y_test, category_names):
    """
    Predicts target data using trained GridSearchCV model. Prints a classification report.
    
    Arguments:
    model: trained CV model
    X_test: test data set to use for prediction
    Y_test: target data set to use for comparison
    
    Returns:
    none
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print GridSearch findings
    print("Best score of CV model = " + model.best_score_)
    print("Best parameters of CV model = " + model.best_params_)
    
    # print classification report for all categories tested
    for i in range(len(category_names)):
        print("Classification report for " + category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
        print("Accuracy = " + str((Y_pred[:,i] == Y_test[:,i]).mean()))

        
def save_model(model, model_filepath):
    """
    Saves model as a pickle file.
    
    Arguments:
    model: trained machine learning model
    model_filepath: destination filepath
    
    Returns:
    none
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

def executeWithoutGridSearch(database_filepath, model_filepath):
    """
    Completes ML process without use of GridSearchCV.
    
    Arguments:
    database_filepath: filepath to database containing data for analysis
    model_filepath: filepath to save model as pickle file
    
    Returns:
    none
    """
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model without GridSearch...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

def executeWithGridSearch(database_filepath, model_filepath):
    """
    Completes ML process with use of GridSearchCV.
    
    Arguments:
    database_filepath: filepath to database containing data for analysis
    model_filepath: filepath to save model as pickle file
    
    Returns:
    none
    """
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model with GridSearch...')
    model = build_CV_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')
        
def main():
    """
    Execute ML training, testing, and saving.
    
    Args:
    [1]: string. Filepath of database to load data from
    [2]: string. Filepath of pickle file to export model
    [3]: (optional) string. indication of whether to use GridSearch
   
    Returns:
    none
    """
    
    # execute ML process without use of GridSearchCV
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        executeWithoutGridSearch(database_filepath, model_filepath)

    # execute ML process with possible use of GridSearchCV
    elif len(sys.argv) == 4:
        database_filepath, model_filepath, usegridsearch = sys.argv[1:]
        if usegridsearch == "usegridsearch":
            executeWithGridSearch(database_filepath, model_filepath)
        else:
            executeWithoutGridSearch(database_filepath, model_filepath)

    # print instructions to user
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
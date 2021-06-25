# import libraries
import sys
import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("MessagesCategories",engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    cols = df.columns[4:].tolist()
    return X, y, cols


def tokenize(text):
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
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [5, 10]
    }
    
    return GridSearchCV(pipeline, param_grid=parameters, n_jobs = 1)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts target data using trained GridSearchCV model. Prints a classification report.
    
    Arguments:
    model: trained CV model
    X_test: test data set to use for prediction
    Y_test: target data set to use for comparison
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    print("Best score of CV model = " + model.best_score_)
    print("Best parameters of CV model = " + model.best_params_)
    
    labels = np.unique(Y_pred)
    conf_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()
    
    print("Labels = " + labels)
    print("Confusion Matrix = ")
    print(conf_mat)
    print("Accuracy = ", accuracy)
    
    for i in range(len(category_names)):
        print("Classification report for " + cols[i])
        print(classification_report(y_test[:,i], y_pred[:,i]))
        print("Accuracy = " + float((y_pred[:,i] == y_test[:,i]).mean()))

def save_model(model, model_filepath):
    """
    Saves model as a pickle file.
    
    Arguments:
    model: trained CV model
    model_filepath: destination filepath
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # database_filepath = 'sqlite:///DisasterResponse.db'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
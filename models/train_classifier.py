# sys
import glob
import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
# natural language toolkit
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# preprocessing
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# pipelines
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
# custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
# train test split
from sklearn.model_selection import train_test_split
# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# metrics
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
# saving model pickle file
import pickle

# download ntlk
# punkt - tokenize
nltk.download('punkt')
# stopwords - stopwords
nltk.download('stopwords')
# lemmatizer
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Read database based on filepath
    @param - database_filepath (str) - absolute file path for .db database
    Returns: X (text feature data), Y (predicting categories) and category names of Y 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', con=engine)
    X = df['message']
    Y = df.loc[:, 'related': ]
    engine.dispose()
    return X, Y, Y.columns

# Variables for tokenize function, and initialisation of Lemmatizer
regex = r"[^A-Za-z]"
url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    '''
    Preprocesses and tokenizes text
    @param - text (str) - string of text data
    Returns - Tokenized array of preprocessed text 
    '''
    # Remove urls
    text = re.sub(url_regex, " ", text)
    # Normalize text
    text = re.sub(regex, " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Stop words and lemmatize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words("english")]
    return words

def build_model():
    '''
    Constructs pipeline by applying:
    - bag of words
    - tfidf transformer
    - wrapping a multi-label classifier to model
    Returns - model pipeline
    '''
    # Initialising model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, n_estimators=10)))
    ])
    return pipeline

def get_all_scores(y_test, y_pred, category_names):
    '''
    Calculates accuracy, precision, recall and f1 score for our model
    @param - y_test (pd.DataFrame/np.array) - test labels
    @param - y_pred (pd.DataFrame/np.array) - model prediction results
    @param - category_names (array) - labels for test set
    Returns - pd.DataFrame of collated accuracy, precision and recall scores
    '''
    prec = []
    recall = []
    f1 = []
    accuracy = []
    support = []
    for i in range(y_test.shape[1]):
        test = y_test.iloc[:, i]
        pred = y_pred[:, i]
        prec.append(precision_score(test,pred))
        recall.append(recall_score(test,pred))
        f1.append(f1_score(test,pred))
        accuracy.append(accuracy_score(test,pred))
        support.append(y_test.iloc[:, i].sum())
    return pd.DataFrame(index=category_names, data={'precision':prec, 'recall':recall, 'f1':f1,\
                                                    'accuracy': accuracy, 'support':support})

def print_weighted(y_test, y_pred, category_names):
    '''
    Prints weighted precision, recall f1 and accuracy
    @param - y_test (matrix) - y_test labels
    @param - y_pred (matrix) - y_pred prediction matrix
    @param - category_names (array) - labels for test set
    '''
    scores = get_all_scores(y_test, y_pred, category_names)
    weighted_precision = (scores['precision'] * (scores['support'] / scores['support'].sum())).sum()
    weighted_recall = (scores['recall'] * (scores['support'] / scores['support'].sum())).sum()
    weighted_f1 = (scores['f1'] * (scores['support'] / scores['support'].sum())).sum()
    weighted_accuracy = (scores['accuracy'] * (scores['support'] / scores['support'].sum())).sum()
    print(f'Weighted precision: {weighted_precision}')
    print(f'Weighted recall: {weighted_recall}')
    print(f'Weighted f1: {weighted_f1}')
    print(f'Weighted accuracy: {weighted_accuracy}')    

    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints model evaluation results
    @param - model - fitted model with training data
    @param - X_test (pd.Series/array) - text words
    @param - Y_test (pd.DataFrame/np.array) - multi-labels for X_test text feature
    @param - category_names (array) - multi-label category names 
    '''
    Y_pred = model.predict(X_test)
    print_weighted(Y_test, Y_pred, category_names)

# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
def save_model(model, model_filepath):
    '''
    Saves model in .pkl format according to filepath
    @param - model - fitted model with training data
    @param - model_filepath (str) - absolute file path to save model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
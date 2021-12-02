# import libraries
import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:].astype('int')
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """This function use the result of best_params_ from GridSearchCV 
        with parameters of {
        'tfidf__use_idf': (True, False),
        'tfidf__smooth_idf': [True, False],
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]}.

        The result of best parameters was 
        {'clf__estimator__min_samples_split': 4,
        'clf__estimator__n_estimators': 100,
        'message_pipe__tfidf__smooth_idf': False,
        'message_pipe__tfidf__use_idf': True,
        'message_pipe__vect__max_df': 0.5,
        'message_pipe__vect__max_features': 5000}
        """

    message_pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=5000)),
        ('tfidf', TfidfTransformer(smooth_idf=False, use_idf=True))
        ])

    # Combining with Pipeline and FeatureUnion to build machine learning pipeline.
    ml_pipeline = Pipeline([
        ('message_pipe', message_pipe),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=4, n_estimators=100, n_jobs=-1)))
        ])

    return ml_pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred_df[col]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)

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
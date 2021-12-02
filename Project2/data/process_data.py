# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    try:
        messages.shape[0] == categories.shape[0]
    except:
        print("Message file length and category file length do not match!!")

    # merge datasets
    df = messages.merge(categories, on='id', how='left')

    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = categories.loc[0, :].apply(lambda x: x.split('-')[0]).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in category_colnames:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    n_duplicates = len(df[df.duplicated(keep=False)])
    print("There are {} duplicates in concatenated dataframe.".format(n_duplicates))
    # drop duplicates
    df = df.drop_duplicates(keep='first')

    # check if there's error in the values of target categories.
    target_df = df.iloc[:, 4:]
    error_cols = []
    for col in target_df.columns:
        if len(target_df[col].unique()) != 2:
            error_cols.append(col)
        else:
            try:
                target_df[col].unique() == [0, 1]
            except:
                error_cols.append(col)
                print("{} : There are values that are not in 0 or 1. Unique values : {}".format(col, target_df[col].unique()))
    
    # remove rows with value of 2 in 'related' column.
    df = df.drop(df[df['related']==2].index, axis=0)

    # check number of duplicates
    n_duplicates = len(df[df.duplicated()])
    if n_duplicates == 0:
        print("Removed duplicates in concatenated dataframe.")
    else:
        print("Still there are duplicates in concatenated dataframe.")

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads and merges dataframes for messages and categories based on id tag
    @param - messages_filepath (str) - path of messages csv
    @param - categories_filepath (str) - path of categories csv
    returns - merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages.drop_duplicates(subset='id')
    # https://stackoverflow.com/questions/37095161/number-of-rows-changes-even-after-pandas-merge-with-left-option
    return messages.merge(categories.drop_duplicates(subset='id'), on='id', how='left')

def clean_data(df):
    '''
    Expands the 'categories' feature and converts into appropriate interger data type
    @param - df (DataFrame) - merged dataframe constructed from load_data() function
    returns - updated dataframe, with expanded values and correct data type
    '''
    categories = df['categories'].str.split(';', expand=True)
    # Extract category column names from first row and update column names
    category_colnames = categories.loc[0, :].str.split('-').str[0]
    categories.columns = category_colnames
    # Convert category values to numbers `0` or `1`
    for column in categories:
        categories[column] = categories[column].str.split('-').str[-1]
        categories[column] = categories[column].astype(int)
        categories[column] = categories[column].apply(lambda x : 1 if x >= 1 else 0)
    # replace df with new category columns
    df.drop('categories', axis=1, inplace=True)
    return pd.concat([df, categories], axis=1)

def save_data(df, database_filename):
    '''
    Saves dataframe to sql
    @param - df (DataFrame) - processed dataframe from clean_data()
    @param - database_filename (str) - .db file name, needs to have .db as extension
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)

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
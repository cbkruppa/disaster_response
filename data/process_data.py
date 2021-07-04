import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets from CSV and merge.
    
    Args:
    messages_filepath: string. Filepath of csv containing message data
    categories_filepath: string. Filepath of csv containing category data
    
    Returns:
    single dataframe containing messages merged with categories
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge the messages and categories datasets using the common id
    categories_nodups = categories.drop_duplicates(subset=['id'])
    df = messages.merge(categories_nodups, left_on='id', right_on='id', how='left')
    
    return df

def clean_data(df):
    """Split the single 'categories' column of the dataframe into separate columns
    for each category.
    
    Args:
    df: Pandas DataFrame. single dataframe containing messages merged with categories
   
    Returns:
    Pandas DataFrame with categories split into separate columns with int values
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # convert column to binary by setting all non-zero values to one
        categories[column] = categories[column].apply(lambda x: 0 if x == 0 else 1)

    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df2 = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df2.drop_duplicates(inplace=True)
    
    return df2


def save_data(df, database_filename):
    """Save dataframe to SQLite database.
    
    Args:
    df: Pandas DataFrame. Dataframe to save to database.
    database_filename: string. Name of database to save to.
   
    Returns:
    none
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')  


def main():
    """Load data from csv of messages and csv of categories. Merge and clean data.
    Save data to database
    
    Args:
    [1]: string. Filepath of CSV containing messages.
    [2]: string. Filepath of CSV containing categories.
    [3]: string. Filepath of database to save to.
   
    Returns:
    none
    """
    
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

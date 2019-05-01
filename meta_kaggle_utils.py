import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(values, title='', xlabel='', ylabel='frequency'):
    """
    Creates and displays simple histogram

    na values are dropped from pandas objects (to not break the hist function)
    """
    # this will only work (and is only needed for) on pandas dataframes. If the type
    #    is something else, it will skip and still plot
    try:
        values = values.dropna()
        values = values.tolist()
    except AttributeError:
        pass

    plt.hist(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def load_kaggle_csv(file_name, index_col='Id', level='normal'):
    """
    Load in one of the kaggle csv files

    By default the number of rows and column names are printed. If print = 'silent'
       nothing is printed. If print = 'verbose' the first 5 rows are printed.
    Args:
        file_name (str): path to the file to load
        index_col (str): name of the column to use as the
           index. '' to create a new index not from a column
        level (str): one of silent, verbose, normal
    """
    # load the file
    df = pd.read_csv(file_name, index_col=index_col)

    # print things based on level
    if level == 'silent':
        pass
    else:
        print('The file contains ' + str(df.shape[0]) + ' rows.')
        print('The table contains the following columns: ')
        print(df.columns.values)
    if level == 'verbose':
        print(df.head())

    return df

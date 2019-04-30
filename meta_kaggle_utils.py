import re
import glob
import boto3
from urllib.request import urlopen
from sys import platform
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import csv
from gensim.parsing.preprocessing import remove_stopwords, strip_numeric, strip_punctuation, strip_short
from gensim.utils import tokenize


def check_platform(aws):
    """
    Determines which platform the code is running on.

    Returns:
        p (str)
    """
    if not aws:
        # if for sure not aws, check which platform
        if platform == 'win32':
            p = 'win'
        if platform == 'darwin':
            p = 'mac'
    else:
        # see if you can connect to aws - assign aws
        p = ''
        url = 'http://169.254.169.254/latest/meta-data'
        try:
            urlopen(url).read()
            p = 'aws'
        except OSError:
            if platform == 'win32':
                p = 'win'
            if platform == 'darwin':
                p = 'mac'

    return p


def load_txt(fname, p, bucket_name='ofi-draper-qa-ue1'):
    """
    Loads a text file from the local path or from an AWS S3 bucket.

    Args:
        fname (str): file path of the text file to be loaded.
        p (str): aws flag
        bucket_name (str): the name of the S3 bucket that the file should be loaded from
    Returns:
        report (string): a string containing the contents of the text file
    """

    if not (p == 'aws'):
        with open(fname, 'r') as file:
            report = file.read()
    else:
        s3_resource = boto3.resource('s3')
        obj = s3_resource.Object(bucket_name, fname)
        report = obj.get()['Body'].read().decode('utf-8')

    return report


def aws_file_list(bucket_name='ofi-draper-qa-ue1'):
    """
    Loads a list of all of the files with a provided AWS header

    Within a provided AWS S3 bucket, this will find all of the files with a given folder name (or other header).
    The header mush match from the root folder of the bucket and be an exact match to the beginnng of the file
    name. It automatically seaches recursively. Wildcard and regular expression searches do not work, use
    find_files to search within a list of files (the output of this function. The command will only return up to
    1000 files at a time, so the ContinuationToken argument is used to make multiple calls getting the matches
    sequentially.

    Args:
        bucket_name (str): the name of the S3 bucket that should be looked in
    Returns:
        files (list of strings): a list containing the full file paths for each file starting with the provided
            folder
    """

    s3 = boto3.client('s3')
    files = []

    kwargs = {'Bucket': bucket_name}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            files.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    return files


def find_files(folder, match, p):
    """
    Searches for files matching a given pattern.

    This replaces glob.glob imports on AWS. Since wildcard searches of S3 buckets don't work, when run with the
    AWS flag, it will search for the match string in the list of files provided, all_files. When AWS is false,
    it will search for files locally matching folder * match * .txt.

    Args:
        folder (str): The folder within which to search
        match (str): the string that is looked for
        p (bool): a binary flag. If true, then we want to search within all_files (likely the output of
            aws_file_list). If false, then it will use glob.glob to search within the provided folder.
    Returns:
        files (list of strings): a list containing the full file paths for each matching file
    """

    if not (p == 'aws'):
        files = glob.glob(folder + '*' + match + '*.txt', recursive=True)
    else:
        files = [f for f in aws_file_list() if match in f and re.search('(10-?[KQ^A])(405*[^A])*(SB*[^A])*_', f)
                 and 'sample' not in f]

    return files


def preprocess_text_word2vec(text):
    """
    """
    text = strip_numeric(text)
    text = strip_punctuation(text)
    text = remove_stopwords(text)
    text = strip_short(text, minsize=3)
    words = list(tokenize(text, lower=True))

    return words


def preprocess_text_sentiment(text):
    """
    takes in text in string form and returns a preprocessed list of words
    Args:
        text (string): I think that this takes in a string that's the text of the document
    Returns:
        words (list[strings]): list of strings of all the words in the document that are not numbers or punctuation

    """
    text = strip_numeric(text)
    text = strip_punctuation(text)
    words = list(tokenize(text, lower=True))

    return words


def get_path(plat):

    if plat == 'aws':
        path = 'data/Company_Filings/parsed'
    elif plat == 'win':
        path = r'//draper.com/services/FS5-Projects/OFI/Company_Filings/*/*/'
    elif plat == 'mac':
        path = '/Volumes/FS5-Projects/OFI/Company_Filings/*/*/'
    else:
        path = ''

    return path


class Tokens(object):
    """
    """
    def __init__(self, files, plat):
            self.files = files
            self.plat = plat

    def __iter__(self):
        for file in self.files:
            text = load_txt(file, self.plat)
            words = preprocess_text_word2vec(text)
            yield words


def setup_dictionary(dictionary_data):
    """
    set up the dictionary of words with their sentiment scores
    Args:
        dictionary_data (string): file path to csv of dictionary of words and their sentiment scores
    Returns:
        sentiment_dictionary (dictionary): python dictionary of dict[word] = sentiment scores
    """
    # set up dictionary
    sentiment_dictionary = {}

    # read from csv of dictionary to make python dictionary:
    with open(dictionary_data) as csv_file:
        dictionary_csv_reader = csv.reader(csv_file, delimiter=',')
        next(dictionary_csv_reader, None)  # skip the headers
        for row in dictionary_csv_reader:
            try:
                float(row[2])
                if len(row) == 3 and -1 < float(row[2]) < 1:  # if the value of the word is between 0 and 1
                    sentiment_dictionary[row[1]] = row[2]
            except ValueError:
                continue

    return sentiment_dictionary


def setup_syntactic_negations(syntactic_negations_data):
    """
    read in data of syntactic negations and put them into a set
    :param syntactic_negations_data: in csv form
    :return: set of negation
    """
    syntactic_negations = set()

    with open(syntactic_negations_data) as csv_file:
        negations_csv = csv.reader(csv_file, delimiter=',')
        for row in negations_csv:
            for word in row:
                syntactic_negations.add(word)

    return syntactic_negations


def setup_diminishers(diminishers_data):
    """
    read in data of syntactic negations and put them into a set
    :param diminishers_data: in csv form
    :return: set of negation
    """
    diminishers = set()

    with open(diminishers_data) as csv_file:
        negations_csv = csv.reader(csv_file, delimiter=',')
        for row in negations_csv:
            for word in row:
                diminishers.add(word)

    return diminishers


def setup_sentiment_analysis(dictionary_data, syntactic_negations_data, diminishers_data):
    """
        sets up sentiment dictionary, syntactic negations, and diminishers

        Args:
            dictionary_data (str): file path of the sentiment dictionary to be loaded.
            syntactic_negations_data (str): file path of the syntactic negations to be loaded.
            diminishers_data (str): file path of the diminishers to be loaded.
        Returns:
            sentiment_dictionary (dictionary): mapping words to scores
            syntactic_negations (set): set of negations (strings)
            diminishers (set): set of diminishers (strings)
        """
    sentiment_dictionary = setup_dictionary(dictionary_data)
    syntactic_negations = setup_syntactic_negations(syntactic_negations_data)
    diminishers = setup_diminishers(diminishers_data)

    return sentiment_dictionary, syntactic_negations, diminishers


def analyze_file_sentiment(sentiment_dictionary, file, plat, syntactic_negations, diminishers):
    text = load_txt(file, plat)
    words = preprocess_text_sentiment(text)
    document = words
    scores = []
    negation_steps = 0
    multiplier = 1
    for i in range(len(document)):
        if negation_steps > 0:
            # stepping away from negation
            negation_steps -= 1
        if document[i] in syntactic_negations:
            negation_steps = 5  # keep scope at 5 to begin with
            multiplier = -1
        elif document[i] in diminishers:
            negation_steps = 5
            multiplier = .2
        elif document[i] in sentiment_dictionary:
            if negation_steps == 0:
                multiplier = 1
            scores.append(multiplier * float(sentiment_dictionary[document[i]]))
    return scores


def analyze_string_sentiment(sentiment_dictionary, word_string, syntactic_negations, diminishers):
    words = preprocess_text_sentiment(word_string)
    scores = []
    negation_steps = 0
    multiplier = 1
    for i in range(len(words)):
        if negation_steps > 0:
            # stepping away from negation
            negation_steps -= 1
        if words[i] in syntactic_negations:
            negation_steps = 5  # keep scope at 5 to begin with
            multiplier = -1
        elif words[i] in diminishers:
            negation_steps = 5
            multiplier = .2
        elif words[i] in sentiment_dictionary:
            if negation_steps == 0:
                multiplier = 1
            scores.append(multiplier * float(sentiment_dictionary[words[i]]))

    # figure out mean and variance of scores
    mean_score = np.mean(scores)
    variance = np.var(scores)
    return mean_score, variance

import re
import os
import nltk
import pickle
import squarify
import pandas as pd
import seaborn as seaborn

from autocorrect import Speller
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from textstat import textstat
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob


def saveObject(obj: object) -> pickle.dump:
    """
    Convert object into binary file, and saved it on disk.
    :param obj: any object
    :return: pickle (dump) file
    """
    new_file = open("tweets.df", 'ab')
    pickle.dump(obj, new_file)
    new_file.close()


def loadObject(obj: pickle) -> pd.DataFrame:
    """
    Load pickle file (binary) and convert it to original form.
    :param obj: any pickle (dump) file.
    :param obj: original object.
    """
    saved_file = open(obj, 'rb')
    list_file = pickle.load(saved_file)
    saved_file.close()
    return list_file


def load(file: object) -> pd.DataFrame:
    """
    Load tweets file.
    :file: csv file path.
    :return: Pandas DataFrame file.
    """
    twitter_file = pd.read_csv(file, delimiter=",")
    file_length = len(twitter_file)

    for i, _ in twitter_file.iterrows():
        print('\rConverting csv to dataframe: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        twitter_file.at[i, 'tweet'] = twitter_file.at[i, 'tweet'].split()
    print('\r', end='')

    return twitter_file


def wordsPerTweet(df: pd.DataFrame) -> dict:
    """
    Count number of words in each lines.
    :param df: pandas DataFrame object.
    :return: (row: tweets count)
    """
    words_in_line = dict()
    for i, _ in df.iterrows():
        words_in_line[i] = len(df.at[i, 'tweet'])

    return words_in_line


def lettersPerTweet(df: pd.DataFrame) -> dict:
    """
    Count number of letters in each line without characters.
    :param df: pandas DataFrame object.
    :return (row: num of letter)
    """
    letters_in_line = dict()
    for i, _ in df.iterrows():
        arr_to_str = ''.join(df.at[i, 'tweet'])
        letters_in_line[i] = len(re.sub('[^a-zA-Z]+', '', arr_to_str))

    return letters_in_line


def averagePerTweet(df: pd.DataFrame) -> dict:
    """
    Calculating the average word length in each tweet
    :param df: pandas DataFrame object.
    :return: (row: average of words)
    """
    words_in_tweet = dict()
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        averege_tweet = round(sum(map(len, filtered_row)) / len(filtered_row), ndigits=3)
        words_in_tweet[i] = averege_tweet

    return words_in_tweet


def countStopWords(df: pd.DataFrame) -> dict:
    """
    Count stop words from the given words list.
    :param df: pandas DataFrame object.
    :return: (row: num of stop words)
    """
    stop_words_in_tweet = dict()
    stop_words = set(stopwords.words('english'))

    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet']]
        filtered_stop_words = len([value for value in filtered_row if value.lower() in stop_words])
        stop_words_in_tweet[i] = filtered_stop_words

    return stop_words_in_tweet


def countNumericLetters(df: pd.DataFrame) -> dict:
    """
    Count the number of numeric characters in every words list.
    :param df: pandas DataFrame object.
    :return: (row: num of numeric chars per tweet)
    """
    numeric_chars = dict()

    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^0-9]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^0-9]+', '', value)) > 0]
        numeric_chars[i] = len(filtered_row)

    return numeric_chars


# noinspection PyTypeChecker
def specialCharacters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count specials chars in each words list.
    :param df: pandas DataFrame object.
    :return: pandas DataFrame that contains number of special chars for each tweet.
    """
    chars = ['!', '"', '#', '$', '%', '&', """'""", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '=', '?', '@',
             '[', ']', '^', '_', '`', '{', '}']
    chars_df = pd.DataFrame(columns=chars)
    file_length = len(df)

    for i, _ in df.iterrows():
        print('\rCounting special characters per tweet: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        for word in df.at[i, 'tweet']:
            for char in word:
                if char in chars_df.columns:
                    chars_df.loc[len(chars_df)] = 0
                    chars_df.at[i, char] += 1
    print('\r', end='')

    return chars_df


def upperCaseCount(df: pd.DataFrame) -> dict:
    """
    Count all upper case words in every tweet.
    :param df: pandas DataFrame object.
    :return: (row: num of uppercase words per tweet)
    """
    upper_case_words = dict()
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^A-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^A-Z]+', '', value)) > 0]
        upper_case_words[i] = len(filtered_row)

    return upper_case_words


# noinspection PyTypeChecker
def wordsCloud(df: pd.DataFrame) -> None:
    """
    Create words cloud with the most common words in the tweets.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """
    concat_all_tweets = ""
    concat_zero_tweets = ""
    concat_one_tweets = ""
    file_length = len(df)

    for i, _ in df.iterrows():
        print('\rMerging all tweets to single string: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        concat_all_tweets += ' '.join([re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet']])
    print('\rMerging all tweets to single string: Done ✓')

    for i, _ in df.iterrows():
        print('\rMerging label [0] tweets into single string: {0}%'.format(round(i / file_length * 100, ndigits=3)),
              end='')
        concat_zero_tweets += ' '.join(
            [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if df.at[i, 'label'] == 0])
    print('\rMerging label [0] tweets into single string: Done ✓')

    for i, _ in df.iterrows():
        print('\rMerging label [1] tweets into single string: {0}%'.format(round(i / file_length * 100, ndigits=3)),
              end='')
        concat_one_tweets += ' '.join(
            [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if df.at[i, 'label'] == 1])
    print('\rMerging label [1] tweets into single string: Done ✓')

    print('\rGenerating clouds, this might take a while...', end='')
    cloud_1 = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(concat_all_tweets)
    cloud_2 = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(concat_zero_tweets)
    cloud_3 = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(concat_one_tweets)
    print('\rGenerating clouds, this might take a while...Done ✓')

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(cloud_1)
    plt.show()

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(cloud_2)
    plt.show()

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(cloud_3)
    plt.show()


# noinspection PyTypeChecker
def treeMapWords(df: pd.DataFrame) -> None:
    """
    Create TreeMap with the most common words in the tweets.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """
    zero_words_vector = []
    one_words_vector = []
    file_length = len(df)

    for i, _ in df.iterrows():
        print('\rMerging label [0] words to single list: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0 and df.at[i, 'label'] == 0]
        zero_words_vector += filtered_row

    print('\rMerging label [0] words to single list: Done ✓')

    for i, _ in df.iterrows():
        print('\rMerging label [1] words to single list: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0 and df.at[i, 'label'] == 1]
        one_words_vector += filtered_row

    print('\rMerging label [1] words to single list: Done ✓')

    print('\rGenerating TreeMap, this might take a while...', end='')
    unique_words_zero = Counter(zero_words_vector)
    most_common_zero = unique_words_zero.most_common(20)
    uniques_zero, sizes_zero = [value[0] for value in most_common_zero], [value[1] for value in most_common_zero]

    unique_words_one = Counter(one_words_vector)
    most_common_one = unique_words_one.most_common(20)
    uniques_one, sizes_one = [value[0] for value in most_common_one], [value[1] for value in most_common_one]
    print('\rGenerating TreeMap, this might take a while... Done ✓')

    squarify.plot(sizes=sizes_zero, label=uniques_zero, alpha=.8)
    plt.axis('off')
    plt.show()

    squarify.plot(sizes=sizes_one, label=uniques_one, alpha=.8)
    plt.axis('off')
    plt.show()


def wordsHistogram(df: pd.DataFrame) -> None:
    """
    Create histogram with words/tweets ratio.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """
    words_per_tweet = wordsPerTweet(df).values()
    plt.hist(x=words_per_tweet, range=(min(words_per_tweet), max(words_per_tweet)), bins=50)
    plt.show()


def lettersHistogram(df: pd.DataFrame) -> None:
    """
    Create histogram with letters/tweets ratio.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """
    letters_length = lettersPerTweet(df).values()
    plt.hist(x=letters_length, range=(min(letters_length), max(letters_length)), bins=100)
    plt.show()


def ratingTweetHistogram(df: pd.DataFrame) -> None:
    """
    Find out how many of the tweets are Positive, Natural, Negative.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """

    def polarity(text):
        return TextBlob(text).sentiment.polarity

    def sentiment(x):
        if x < 0:
            return 'Negative'
        elif x == 0:
            return 'Natural'
        else:
            return 'Positive'

    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = ' '.join(filtered_row)

    # Polarity range (-1.0,1.0) distribution
    df['rating'] = df['tweet'].apply(lambda x: polarity(x))
    df['rating'].hist()
    plt.show()

    # Polarity category distribution
    df['rating'] = df['rating'].map(lambda x: sentiment(x))
    plt.bar(df['rating'].value_counts().index, df['rating'].value_counts())
    plt.show()


def partOfSpeechTweet(df: pd.DataFrame) -> None:
    """
    Plot bar with the most common part-of-speech words.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """

    def part_of_speech(tweet: list) -> list:
        """
        For every row in DataFrame, tokenize, split to POS and leave POS tags only.
        :param tweet: tokenized tweet.
        :return: list of pos tokens in row.
        """
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in tweet if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]

        pos_in_row = nltk.pos_tag(filtered_row)
        tags_in_row = list(map(list, zip(*pos_in_row)))[1]

        return tags_in_row

    print("\rTokenizing every tweet to Part-Of-Speech...", end='')
    df['tags'] = df['tweet'].apply(lambda x: part_of_speech(x))
    print("\rTokenizing every tweet to Part-Of-Speech...Done ✓")

    print("\rCounting Part-Of-Speech tokens for all tweets...", end='')
    tags = [word for row in df['tags'] for word in row]
    counter = Counter(tags)
    print("\rCounting Part-Of-Speech tokens for all tweets...Done ✓")

    print("\rCreating Bar plot...", end='')
    x, y = list(map(list, zip(*counter.most_common(7))))
    seaborn.barplot(x=y, y=x)
    plt.legend([
        "ADJ - Adjective", "ADP - Adposition", "ADV - Adverb", "CONJ - Conjunction", "DET - Determiner"
        , "NN - noun", "NUM - Numeral", "PRT - Particle", "PRON - Pronoun", "VERB - Verb"
    ])
    print("\rCreating Bar plot...Done ✓")
    plt.show()


def readableTweet(df: pd.DataFrame) -> None:
    """
    Find out how easy the tweets are to read.
    :param df: pandas DataFrame object.
    :return: None (plt images is being showed).
    """
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'sentence'] = ' '.join(filtered_row)

    df['sentence'].apply(lambda x: textstat.flesch_reading_ease(x)).hist(bins=100)
    plt.xlim([-100, 100])
    plt.legend([" Under -30 [Difficult To Read]\n Under (-31) to (-60) [Normal To Read]\nAbove -60 [Easy To Read]"])
    plt.show()


def convertToLower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every capital letter/word will be converted to lower case.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value.lower()) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    return df


def removeSpecialChars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every special character will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    return df


def removeStopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every stop word will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    stop_words = list(stopwords.words('english'))
    for i, _ in df.iterrows():
        filtered_row = [value for value in df.at[i, 'tweet'] if value not in stop_words]
        df.at[i, 'tweet'] = filtered_row

    return df


# noinspection PyTypeChecker
def removeRareWords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that rare word will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    words_vector = []
    file_length = len(df)

    for i, _ in df.iterrows():
        print('\rMerging all words to single list: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0 and df.at[i, 'label'] == 0]
        words_vector += filtered_row

    print('\rMerging all words to single list: Done ✓')
    unique_words = Counter(words_vector)
    most_uncommon = unique_words.most_common()[-100:]
    print("100 most uncommon words:\n", most_uncommon)

    return df


# noinspection PyTypeChecker
def fixSpellingError(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that rare word will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    file_length = len(df)
    check_spell = Speller(lang='en', fast=True)

    for i, _ in df.iterrows():
        print('\rLooking for misspelled words and fixing them: {0}%'.format(round(i / file_length * 100, ndigits=3)),
              end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', check_spell(value)) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    print('\rLooking for misspelled words and fixing them: Done ✓')

    return df


file_path = os.getcwd() + "\\tweets.df"
print("\rChecking if Pickle file is already exist...", end='\n')

if not os.path.exists(file_path):
    dataframe = load("trainTwitter.csv")
    saveObject(dataframe)
    print("\rNo file is stored, Creating a new one...Done ✓", end='\n')

dataframe = loadObject(file_path)
print("\rLoading stored file...Done ✓", end='\n')


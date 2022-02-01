import os
import re
import time
import math
import pickle
import nltk
import numpy as np
import pandas as pd
from scipy import spatial
from gensim import models
from gensim import corpora
import textblob as textblob

from autocorrect import Speller
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs


def saveObject(obj: object, f_name: str) -> pickle.dump:
    """
    Convert object into binary file, and saved it on disk.
    :param f_name: custom file name.
    :param obj: any object
    :return: pickle (dump) file
    """
    new_file = open("{0}".format(f_name), 'ab')
    pickle.dump(obj, new_file)
    new_file.close()


def loadObject(obj: pickle) -> pd.DataFrame:
    """
    Load pickle file (binary) and convert it to original form.
    :param obj: any pickle (dump) file.
    :param obj: original object.
    """
    saved_file = None
    try:
        saved_file = open(obj, 'rb')
    except FileNotFoundError:
        print('[{0}] No such file or Directory...'.format(obj))
        exit(1)

    list_file = pickle.load(saved_file)
    saved_file.close()
    return list_file


def convertToLower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every capital letter/word will be converted to lower case.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """

    dispatchMessage()['convertToLower'](True)
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value.lower()) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    dispatchMessage()['convertToLower'](False)
    return df


def removeSpecialChars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every special character will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """

    dispatchMessage()['removeSpecialChars'](True)
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    dispatchMessage()['removeSpecialChars'](False)
    return df


def removeStopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every stop word will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """
    dispatchMessage()['removeStopwords'](True)
    stop_words = list(stopwords.words('english'))
    for i, _ in df.iterrows():
        filtered_row = [value for value in df.at[i, 'tweet'] if value not in stop_words]
        df.at[i, 'tweet'] = filtered_row

    dispatchMessage()['removeStopwords'](False)
    return df


# noinspection PyTypeChecker
def fixSpellingError(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that rare word will be removed.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """

    dispatchMessage()['fixSpellingError'](True)
    check_spell = Speller(lang='en', fast=True)
    file_length = len(df)

    for i, _ in df.iterrows():
        print('\rLooking for misspelled words and fixing them: {0}%'.format(round(i / file_length * 100, ndigits=3)),
              end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', check_spell(value)) for value in df.at[i, 'tweet'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'tweet'] = filtered_row

    print('\rLooking for misspelled words and fixing them: Done ✓')
    dispatchMessage()['fixSpellingError'](False)

    return df


def nltk_tokenize(text) -> list:
    """
    Tokenize text using NLTK
    :param text: Tweet (String).
    :return: list of tokenized words.
    """
    tokens = nltk.word_tokenize(text)
    return tokens


def textblob_spell_checker(tweet: str) -> list:
    """
    Spell checker using TextBlob.
    :param tweet: String
    :return: list of corrected misspelled words.
    """
    corrected_tweet = textblob.TextBlob(tweet).correct()
    return nltk_tokenize(str(corrected_tweet))


def nltk_lemmatization(tweet: str) -> list:
    """
    Lemmatization using NLTK.
    :param tweet: String
    :return: list of lemmatized words.
    """
    tokenized_tweet = nltk_tokenize(tweet)
    lemmatized_words = []
    for word in tokenized_tweet:
        lemmatized_words += [nltk.WordNetLemmatizer().lemmatize(word)]
    return lemmatized_words


def nltk_stemming(tweet: str) -> list:
    """
    Stemming using NLTK.
    :param tweet: String
    :return: list of stemmed words.
    """
    tokenized_tweet = nltk_tokenize(tweet)
    stemmed_words = []
    for word in tokenized_tweet:
        stemmed_words += [nltk.PorterStemmer().stem(word)]
    return stemmed_words


def nlp_manipulations(df: pd.DataFrame) -> dict:
    """
    Perform manipulations on Dataframe based on chosen function.
    :param df: pandas Dataframe.
    :return: manipulated dataframe.
    """

    def lemmatization(n_rows: int = len(df)) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        dispatchMessage()['lemmatization'](True)

        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'tweet'])
            lemmatized_tweet = nltk_lemmatization(tweet)

            if lemmatized_tweet != list(df.at[i, 'tweet']):
                df.at[i, 'tweet'] = lemmatized_tweet

        end_time = time.time()
        dispatchMessage()['lemmatization'](False)
        print("Running Time: {0:.6f} Seconds.".format(end_time - start_time))

    def stemming(n_rows: int = len(df)) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        dispatchMessage()['stemming'](True)

        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'tweet'])
            stemmed_tweet = nltk_stemming(tweet)

            if stemmed_tweet != list(df.at[i, 'tweet']):
                df.at[i, 'tweet'] = stemmed_tweet

        end_time = time.time()
        dispatchMessage()['stemming'](False)
        print("Running Time: {0:.6f} Seconds.".format(end_time - start_time))

    return {
        'lemmatization': lemmatization,
        'stemming': stemming,
    }


def dispatchMessage() -> dict:
    """
    Print custom message for specific function with Dispatch Dictionary Method.
    :return: None
    """

    def message_convertToLower(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being converted to lower chars...", end='')
        else:
            print("\rTokens are being converted to lower chars...✓", end='\n')

    def message_removeSpecialChars(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Special characters are being removed from tokens...", end='')
        else:
            print("\rSpecial characters are being removed from tokens...✓", end='\n')

    def message_removeStopwords(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Stop words are being removed from tokens...", end='')
        else:
            print("\rStop words are being removed from tokens...✓", end='\n')

    def message_fixSpellingError(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being analyzed and fixed if spelling error is occurred...", end='')
        else:
            print("\rTokens are being analyzed and fixed if spelling error is occurred...✓", end='\n')

    def lemmatization(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being lemmatized...", end='')
        else:
            print("\rTokens are being lemmatized...✓", end='\n')

    def stemming(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being stemmed......", end='')
        else:
            print("\rTokens are being stemmed...✓", end='\n')

    def fill_matrix(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Filling Bag of Words matrix......", end='')
        else:
            print("\rFilling Bag of Words matrix......✓", end='\n')

    def tf_idf_matrix(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Filling TFIDF Matrix...", end='')
        else:
            print("\rFilling TFIDF Matrix...✓", end='\n')

    def cosine_sim(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Calculating Cosine Similarity...", end='')
        else:
            print("\rCalculating Cosine Similarity...✓", end='\n')

    return {
        'convertToLower': message_convertToLower,
        'removeSpecialChars': message_removeSpecialChars,
        'removeStopwords': message_removeStopwords,
        'fixSpellingError': message_fixSpellingError,
        'lemmatization': lemmatization,
        'stemming': stemming,
        'fill_matrix': fill_matrix,
        'tf_idf_matrix': tf_idf_matrix,
        'cosine_similarity': cosine_sim

    }


def preprocessing(df_file: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing the dataframe before using it to manipulations.
    :param df_file: pandas dataframe.
    :return: pd.DataFrame.
    """
    df_file = convertToLower(df_file)
    df_file = removeSpecialChars(df_file)
    df_file = removeStopwords(df_file)
    df_file = fixSpellingError(df_file)

    nlp_manipulations(df_file)['lemmatization']()
    nlp_manipulations(df_file)['stemming']()

    return df_file


def create_dataframe(n_rows: int) -> pd.DataFrame:
    """
    Create dataframe from local csv file.
    :param n_rows: number of rows to be taken from the csv file
    :return: pd.Dataframe.
    """
    dataframe = pd.read_csv('tweets.csv', delimiter=",")[:n_rows]
    f_len = len(dataframe)

    for i, _ in dataframe.iterrows():
        print('\rConverting csv to dataframe: {0:.3f}%'.format(i / f_len * 100), end='')
        dataframe.at[i, 'tweet'] = dataframe.at[i, 'tweet'].split()
    print('All tweets tokenized...Done ✓\r', end='')

    return dataframe


def start(n_rows: int) -> pd.DataFrame:
    pickle_file = os.getcwd() + "\\tweets.pl"
    pp_pickle_file = os.getcwd() + "\\pp_tweets.pl"
    print("\rChecking if Pickle file is already exists...", end='\n')

    if not os.path.exists(pickle_file):
        tweets_df = create_dataframe(n_rows)
        saveObject(obj=tweets_df, f_name='tweets.pl')
        print("\rCreating new pickle file...Done ✓", end='\n')

    dataframe = loadObject(pickle_file)
    print("\rLoading pickle file...Done ✓", end='\n')

    print("\rChecking if Preprocessed Pickle file is already exists...", end='\n')
    if not os.path.exists(pp_pickle_file):
        pp_tweets_df = preprocessing(dataframe)
        saveObject(obj=pp_tweets_df, f_name='pp_tweets.pl')
        print("\rPreprocessing dataframe...Done ✓", end='\n')

    dataframe = loadObject(pp_pickle_file)
    print("\rLoading preprocessed pickle file...Done ✓", end='\n')

    return dataframe


class BagOfWords:
    def __init__(self, dataframe: pd.DataFrame):
        self.data_ = dataframe
        self.words = None
        self.docs = None
        self.df = None
        self.c_size = None
        self.r_size = None

    def __repr__(self):
        return "Bag Of Words Object"

    def set_data(self) -> None:
        """
        Build dataframe for boolean matrix.
        :return: None
        """
        # Reload all words in tweets file.
        self.words = list(set([word for i, row in self.data_.iterrows() for word in row['tweet']]))

        # Generate columns names for dataframe.
        self.docs = ['Doc #{0}'.format(i) for i in range(len(self.data_))]

        # set size for columns and rows.
        self.c_size = len(self.words)
        self.r_size = len(self.docs)

        # Initialize dataframe.
        self.df = pd.DataFrame(np.zeros((self.r_size, self.c_size)))

        # Set names for columns and rows.
        self.df.columns = self.words
        self.df.index = self.docs

    def fill_matrix(self):
        start_time = time.time()

        file_length = len(self.data_)
        dispatchMessage()['fill_matrix'](True)
        for i, row in self.data_.iterrows():
            # noinspection PyTypeChecker
            print('\rFilling Bag Of Words Matrix...: {0}%'.format(round(i / file_length * 100, ndigits=3)), end='')
            for word in self.words:
                if word in row['tweet']:
                    self.df.at['Doc #{0}'.format(i), word] = row['tweet'].count(word)
        dispatchMessage()['fill_matrix'](False)

        end_time = time.time()
        print("Time to build initialize Bag of Words Matrix:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    def query_vector(self, query: str) -> pd.Series:
        """
        Create query vector with bow method.
        :param query: string of words
        :return: pd.Series of bow vector
        """
        # split query into tokens of words
        words_list = preprocess_query(query).split()

        # create empty vectors for calculations
        quantity_vector = dict()

        # iterate over all words and calculate quantity of words
        for word in self.words:
            if word in words_list:
                quantity_vector[word] = words_list.count(word)
            else:
                quantity_vector[word] = 0.0
        return pd.Series(quantity_vector).sort_values(ascending=False)


class TFIDF:
    def __init__(self, bow_df: pd.DataFrame, words_list: list):
        self.df = pd.DataFrame.copy(bow_df)
        self.c_size = len(self.df.columns)
        self.r_size = len(self.df)
        self.words = words_list

    def __repr__(self):
        return "Term Frequency – Inverse Document Frequency Object\n\
               Rows (Documents): {0}\n\
               Columns (Unique Words): {1}".format(self.r_size, self.c_size)

    def fill_matrix(self):
        """
        Create Matrix with tf_idf representation.
        :return: Matrix with values calculated by tf_idf method.
        """
        dispatchMessage()['tf_idf_matrix'](True)
        start_time = time.time()

        row_counter = 0
        dispatchMessage()['fill_matrix'](True)
        for i in range(self.r_size):  # iterate over rows
            print('\rFilling TFIDF Matrix...: {0}%'.format(round(row_counter / self.r_size * 100, ndigits=6)), end='')
            for word in self.words:  # iterate over columns
                self.df.at['Doc #{0}'.format(i), word] = self.calc_tf_idf_td(word, row_counter)
            row_counter += 1
        dispatchMessage()['fill_matrix'](False)

        end_time = time.time()
        dispatchMessage()['tf_idf_matrix'](False)
        print("Time to build initialize TFIDF Matrix:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    def calc_tf_td(self, word: str, document_id: int) -> float:
        """
        Calculate tf_df of word in specific document.
        :param word: string word.
        :param document_id: Int of document.
        :return: float number represents tf_td calculation.
        """
        try:
            tf_td = self.df.at["Doc #{0}".format(document_id), word] / len(self.df.loc["Doc #{0}".format(document_id)])
            if tf_td > 0:
                return 1 + math.log10(tf_td)
            else:
                return 0.0
        except KeyError as e:
            print("[{0}] Does Not Exist!".format(e.args[0]))
            exit(1)

    def calc_idf_t(self, word: str) -> float:
        """
        Number of documents that contains (word) at least once (no repeats).
        wrap result log10(1/dft)
        :param word:
        :return: float number represents idf_t calculation.
        """
        size = self.r_size
        column_vector = 0
        try:
            for row in self.df[word]:
                if row > 0.0:
                    column_vector += 1
            return math.log10(size / column_vector) if column_vector > 0 else math.log10(1 / 1)

        except KeyError as e:
            print("[{0}] Does Not Exist!".format(e.args[0]))
            exit(1)

    def calc_tf_idf_td(self, word: str, document_id: int) -> float:
        """
        Calculate tf_idf for specific word.
        :param word: string token
        :param document_id: Int of document
        :return: float number represents tf_idf calculation
        """
        return abs((self.calc_idf_t(word) * self.calc_tf_td(word, document_id)))

    def query_vector(self, query: str) -> pd.Series:
        """
        Create query vector with tf_idf method.
        :param query: string of words
        :return: pd.Series of tf_idf vector
        """
        # WARNING: idf_t = 1 (will not be calculated, number of documents is only 1 for query vector)
        # split query into tokens of words
        words_list = preprocess_query(query).split()

        # create empty vectors for calculations
        quantity_vector = dict()
        tf_td_vector = dict()

        # iterate over all words and calculate quantity of words
        for word in self.words:
            if word in words_list:
                quantity_vector[word] = words_list.count(word)
            else:
                quantity_vector[word] = 0.0

        # iterate over all words and calculate tf_td of words
        for word in self.words:
            try:
                tc_td = quantity_vector[word]
                if tc_td > 0:
                    tf_td_vector[word] = 1 + math.log10(tc_td)
                else:
                    tf_td_vector[word] = 0.0
            except KeyError as e:
                print("[{0}] Does Not Exist!".format(e.args[0]))
                exit(1)

        return pd.Series(tf_td_vector).sort_values(ascending=False)


def cosine_similarity(d_vector: pd.Series, q_vector: pd.Series) -> float:
    """
    Return similarity between 2 vectors
    :param d_vector: document vector
    :param q_vector: query vector
    :return: float number represents similarity
    """
    numerator = 0
    denominator_d = 0
    denominator_q = 0
    for i in range(len(q_vector)):
        numerator += (q_vector[i] * d_vector[i])
        denominator_d += math.pow(d_vector[i], 2)
        denominator_q += math.pow(q_vector[i], 2)

    return round(numerator / math.sqrt((denominator_d * denominator_q)), ndigits=4)


def rank(obj: BagOfWords or TFIDF, query: str) -> dict:
    """
    :param obj:
    :param query:
    :return:
    """
    start_time = time.time()

    size = len(obj.df)
    size_count = 0
    rank_dict = dict()
    q_vector = obj.query_vector(preprocess_query(query))
    for i, row in obj.df.iterrows():
        print('\rRanking Documents For Query...: {0}%'.format(round(size_count / size * 100, ndigits=6)), end='')
        rank_dict['Rank #{0}'.format(i)] = cosine_similarity(d_vector=row, q_vector=q_vector)
        size_count += 1
    print('Ranking Documents For Query......Done ✓\r', end='')

    end_time = time.time()
    print("Time to rank documents by query:")
    print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    return pd.Series(rank_dict).sort_values(ascending=False)


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key

    return "Key Doesn't Exist"


def sklearn_tf_idf(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Use SkLearn to create tf_idf matrix.
    :param dataset: pd.Dataframe
    :return: tf_idf matrix.
    """
    dispatchMessage()['tf_idf_matrix'](True)
    start_time = time.time()

    def tokens_to_string(df: pd.DataFrame) -> list:
        """
        Convert list of tokens (document words) into string.
        :param df: pd.Dataframe
        :return: list of tweets.
        """
        string_tweets = []
        for row in df:
            string_tweets.append(' '.join(row))
        return string_tweets

    dataset_dictionary = list(set([word for w in dataset for word in w]))
    dataset = tokens_to_string(dataset)

    tf_idf_vectorizer = TfidfVectorizer(analyzer='word', vocabulary=dataset_dictionary)
    tfidf_matrix = tf_idf_vectorizer.fit_transform(dataset)
    tfidf_columns_names = tf_idf_vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), index=['Doc #{0}'.format(i) for i in range(len(dataset))],
                            columns=tfidf_columns_names)
    end_time = time.time()
    dispatchMessage()['tf_idf_matrix'](False)
    print("Time to build initialize SKLearn TFIDF Matrix:")
    print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    return tfidf_df


def sklearn_cosine_sim(vec_1: list, vec_2: list, print_time: bool) -> list:
    """
    Calculate cosine similarity between 2 lists (vectors)
    :param vec_1: numeric list
    :param vec_2: numeric list
    :param print_time: print time duration of process
    :return: calculated similarity (float)
    """
    start_time = time.time()

    result = cs([vec_1], [vec_2])[0]

    end_time = time.time()
    if print_time:
        print("Time to calculate cosine similarity between 2 vectors:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    return result


def gensim_tf_idf(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create tf_idf matrix with gensim library
    :param dataset: pd.Dataframe
    :return: tf_idf matrix.
    """
    dispatchMessage()['tf_idf_matrix'](True)
    start_time = time.time()

    tokenized_vector = pd.Series(dataset)
    tf_idf_list = []

    dictionary = corpora.Dictionary(tokenized_vector)

    # fill dictionary [index-->word] (id2token attribute)
    for index in dictionary.token2id.values():
        dictionary.id2token[index] = get_key(dictionary.token2id, index)

    # calculate tf_idf to all documents
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_vector]
    tfidf_model = models.TfidfModel(corpus)

    for row in dataset:
        vec_bow = dictionary.doc2bow(row)
        tfidf_v = tfidf_model[vec_bow]

        tf_idf_list.append([(dictionary.id2token[tfidf_v[i][0]], tfidf_v[i][1]) for i in range(len(tfidf_v))])

    # create new dataframe and fill cells with the tf-idf values
    tf_idf_df = pd.DataFrame(np.zeros((len(dataset), len(dictionary))), columns=dictionary.token2id.keys(),
                             index=['Doc #{0}'.format(i) for i in range(len(dataset))])
    row_index = 0
    for row in tf_idf_list:
        for vec in row:
            tf_idf_df.at['Doc #{0}'.format(row_index), vec[0]] = vec[1]
        row_index += 1

    end_time = time.time()
    dispatchMessage()['tf_idf_matrix'](False)
    print("Time to build Gensim TFIDF Matrix:")
    print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))
    return tf_idf_df


def scipy_cosine_sim(vec_1: pd.Series, vec_2: pd.Series, print_time: bool) -> float:
    """
    Calculate cosine similarity between 2 vectors
    :param vec_1: pd.Series
    :param vec_2: pd.Series
        :param print_time: print time duration of process
    :return: calculated similarity (float)
    """
    start_time = time.time()

    result = 1 - spatial.distance.cosine(vec_1, vec_2)

    end_time = time.time()
    if print_time:
        print("Time to calculate cosine similarity between 2 vectors:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    return result


def start_libraries(cosine_sim_model: str):
    """
    Run Ranking documents models from pre made libraries.
    :return: sorted ranked list from gensim/sklearn model.
    """
    start_time = time.time()

    preprocessed_df = loadObject(obj="pp_tweets.pl")['tweet']
    tfidf_df = loadObject(obj="tfidf_df.pl")
    size = len(preprocessed_df)
    ranks = []

    def sklearn(query: str) -> pd.Series:
        sklearn_model = sklearn_tf_idf(preprocessed_df).sort_index(axis=1)
        query_v = tfidf_df.query_vector(preprocess_query(query)).sort_index()
        for i in range(size):
            if cosine_sim_model == "sklearn":
                ranks.append(sklearn_cosine_sim(sklearn_model.iloc[i], query_v, False))
            elif cosine_sim_model == "scipy":
                ranks.append(scipy_cosine_sim(sklearn_model.iloc[i], query_v, False))
        return pd.Series(ranks, index=["Doc #{0}".format(i) for i in range(size)]).sort_values(ascending=False)

    def gensim(query: str) -> pd.Series:
        gensim_model = gensim_tf_idf(preprocessed_df).sort_index(axis=1)
        query_v = tfidf_df.query_vector(preprocess_query(query)).sort_index()
        for i in range(size):
            if cosine_sim_model == "sklearn":
                ranks.append(sklearn_cosine_sim(gensim_model.iloc[i], query_v, False))
            elif cosine_sim_model == "scipy":
                ranks.append(scipy_cosine_sim(gensim_model.iloc[i], query_v, False))
        return pd.Series(ranks, index=["Doc #{0}".format(i) for i in range(size)]).sort_values(ascending=False)

    end_time = time.time()
    print("Time to rank documents by query:")
    print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))

    return {
        'sklearn': sklearn,
        'gensim': gensim
    }


def preprocess_query(query: str) -> str:
    """
    Preprocess query for by nltk tools.
    :param query: String
    :return: processed string (query) using nltk tools.
    """
    temp_df = pd.DataFrame(data=query, index=[0], columns=['tweet'])
    temp_df.at[0, 'tweet'] = query.split(' ')
    preprocessed_df = preprocessing(temp_df)

    return ' '.join(preprocessed_df.at[0, 'tweet'])


# print(start_libraries()['sklearn'](query="blog fun run"))
# print(start_libraries()['gensim'](query="blog fun run"))

# bow = BagOfWords(dataframe=start(1000))
# bow.set_data()
# bow.fill_matrix()
# saveObject(obj=bow,f_name="bow_df.pl")
# bw = loadObject(obj="bow_df.pl")
# tfidf = TFIDF(bw.df, bw.words)
# tfidf.fill_matrix()
# saveObject(obj=tfidf, f_name="tfidf_df.pl")
# bw = loadObject(obj="bow_df.pl")
# tfidf = loadObject(obj="tfidf_df.pl")
# print(rank(obj=tfidf, query="blog lobg and stupid"))
# print(rank(obj=bw, query="blog lobg and stupid"))

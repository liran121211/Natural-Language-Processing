import os
import re
import time
import pickle
import nltk
import numpy as np
import pandas as pd
import textblob as textblob
from autocorrect import Speller
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, rand_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups as sklearn_dataset
from sklearn.metrics import silhouette_score
import collections


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
        filtered_row = [re.sub('[^a-zA-Z]+', '', value.lower()) for value in df.at[i, 'data'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'data'] = filtered_row

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
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'data'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'data'] = filtered_row

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
        filtered_row = [value for value in df.at[i, 'data'] if value not in stop_words]
        df.at[i, 'data'] = filtered_row

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

    rows_n = 0
    for i, _ in df.iterrows():
        print('\rLooking for misspelled words and fixing them: {0}%'.format(
            round(rows_n / int(file_length) * 100, ndigits=3)),
            end='')
        filtered_row = [re.sub('[^a-zA-Z]+', '', check_spell(value)) for value in df.at[i, 'data'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'data'] = filtered_row
        rows_n += 1

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
            tweet = ' '.join(df.at[i, 'data'])
            lemmatized_tweet = nltk_lemmatization(tweet)

            if lemmatized_tweet != list(df.at[i, 'data']):
                df.at[i, 'data'] = lemmatized_tweet

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
            tweet = ' '.join(df.at[i, 'data'])
            stemmed_tweet = nltk_stemming(tweet)

            if stemmed_tweet != list(df.at[i, 'data']):
                df.at[i, 'data'] = stemmed_tweet

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

    def sklearn_clustering(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Building SKLearn Clustering Model...\n", end='')
        else:
            print("\rBuilding SKLearn Clustering Model...✓\n", end='\n')

    return {
        'convertToLower': message_convertToLower,
        'removeSpecialChars': message_removeSpecialChars,
        'removeStopwords': message_removeStopwords,
        'fixSpellingError': message_fixSpellingError,
        'lemmatization': lemmatization,
        'stemming': stemming,
        'fill_matrix': fill_matrix,
        'sklearn_clustering': sklearn_clustering,
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
    :param n_rows: number of rows (int)
    :return: pd.Dataframe.
    """
    tuple_data = sklearn_dataset(subset='all',
                                 categories=['talk.politics.misc', 'talk.religion.misc', 'comp.graphics', 'sci.space'],
                                 remove=(), download_if_missing=True, return_X_y=True)
    f_len = len(tuple_data[0][:n_rows])
    np_data = np.array((tuple_data[0][:n_rows], tuple_data[1][:n_rows])).transpose()
    dataframe = pd.DataFrame(data=np_data, columns=['data', 'class'],
                             index=['Doc #{0}'.format(x) for x in range(f_len)])

    row_n = 0
    for i, _ in dataframe.iterrows():
        print('\rTokenizing documents...: {0:.3f}%'.format(row_n / int(f_len) * 100), end='')
        dataframe.at[i, 'data'] = dataframe.at[i, 'data'].split()
        row_n += 1
    print('Tokenizing documents......Done ✓\r', end='')
    return dataframe


def start(n_rows: int) -> pd.DataFrame:
    pickle_file = os.getcwd() + "\\sklearn_dataframe.pl"
    pp_pickle_file = os.getcwd() + "\\pp_sklearn_dataframe.pl"
    print("\rChecking if Pickle file is already exists...", end='\n')

    if not os.path.exists(pickle_file):
        sklearn_df = create_dataframe(n_rows)
        saveObject(obj=sklearn_df, f_name='sklearn_dataframe.pl')
        print("\rCreating new pickle file...Done ✓", end='\n')

    dataframe = loadObject(pickle_file)
    print("\rLoading pickle file...Done ✓", end='\n')

    print("\rChecking if Preprocessed Pickle file is already exists...", end='\n')
    if not os.path.exists(pp_pickle_file):
        pp_sklearn_df = preprocessing(dataframe)
        saveObject(obj=pp_sklearn_df, f_name='pp_sklearn_dataframe.pl')
        print("\rPreprocessing dataframe...Done ✓", end='\n')

    dataframe = loadObject(pp_pickle_file)
    print("\rLoading preprocessed pickle file...Done ✓", end='\n')

    return dataframe


def sklearn_clustering(dataset: pd.DataFrame, matrix_model: str, classifier_model: str, k: int,
                       distance_mode: str = None, linkage: str = None, metrics_: str = None) -> None:
    """
    Use SKLearn to create clusters.
    :param linkage: Determines which distance to use between sets of observation.
    :param distance_mode: [“Euclidean”, “l1”, “l2”, “Manhattan”, “Cosine”, “Precomputed”]
    :param dataset: pd.Dataframe
    :param matrix_model: TFIDF/BOW Matrix.
    :param classifier_model: KMeans/HAC.
    :param k: number of clusters.
    """
    dispatchMessage()['sklearn_clustering'](True)
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

    dataset_reformatted = tokens_to_string(dataset['data'])

    X_train, X_test, y_train, y_test = train_test_split(dataset_reformatted, dataset['class'], test_size=0.30,
                                                        shuffle=True)

    # Avoid use before assignment
    clf = None
    tfidf_data_train, tfidf_data_test, bow_data_train, bow_data_test = [], [], [], []

    if matrix_model == 'TFIDF':
        vectorizer = TfidfVectorizer()
        tfidf_data_train = vectorizer.fit_transform(X_train)

    elif matrix_model == 'BOW':
        vectorizer = CountVectorizer()
        bow_data_train = vectorizer.fit_transform(X_train)

    else:
        raise ValueError('Matrix model does not exist.')

    if classifier_model == 'KMeans':
        clf = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        clf.fit(tfidf_data_train)

        if metrics_ == "Purity":
            y_true = clf.fit_predict(vectorizer.fit_transform(X_test))
            contingency_matrix = metrics.cluster.contingency_matrix(y_test, y_true)
            print("Purity Metric: ", np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
            return None

        if metrics_ == "RAND":
            y_true = clf.fit_predict(vectorizer.fit_transform(X_test))
            print("RAND Metric: ", rand_score(y_test, y_true))
            return None

        clustering = collections.defaultdict(list)
        for idx, label in enumerate(clf.labels_):
            clustering[label].append(idx)

        for key, value in clustering.items():
            print("Cluster #{0} Contains Rows --->".format(key), end='')
            for index in range(len(value)):
                print("[{0}] ".format(index), end='')
            print('\n')

        print("Top Terms Per Cluster:")
        order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print('\n')

        # Check optimal (K) for clusters.
        # sum_of_squared_distances = []
        # K = range(1, 30)
        # for k in K:
        #     k_means = KMeans(n_clusters=k)
        #     k_means = k_means.fit(tfidf_data_train)
        #     sum_of_squared_distances.append(k_means.inertia_)
        #     print(sum_of_squared_distances)
        # plt.plot(K, sum_of_squared_distances, 'bx-')
        # plt.xlabel('Optimal (K)')
        # plt.ylabel('Sum Of Squared Distances')
        # plt.title('Elbow Method For Optimal k')
        # plt.show()

        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        # sil = []
        # k_max = 30
        # for k in range(2, k_max + 1):
        #     k_means = KMeans(n_clusters=k).fit(tfidf_data_train)
        #     labels = k_means.labels_
        #     sil.append(silhouette_score(tfidf_data_train, labels, metric='euclidean'))
        #     print(sil)
        # plt.plot(range(1, k_max), sil, 'bx-')
        # plt.xlabel('Optimal (K)')
        # plt.ylabel('Sum Of Squared Distances')
        # plt.title('Silhouette Method For Optimal k')
        # plt.show()

    elif classifier_model == 'HAC':
        clf = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        clf.fit(tfidf_data_train)

        if metrics_ == "Purity":
            y_true = clf.fit_predict(vectorizer.fit_transform(X_test))
            contingency_matrix = metrics.cluster.contingency_matrix(y_test, y_true)
            print("Purity Metric: ", np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
            return None

        if metrics_ == "RAND":
            y_true = clf.fit_predict(vectorizer.fit_transform(X_test))
            print("RAND Metric: ", rand_score(y_test, y_true))
            return None

        # number of clusters
        model_agglomerative_clustering = AgglomerativeClustering(n_clusters=k, affinity=distance_mode,
                                                                 linkage=linkage)
        model_agglomerative_clustering.fit(tfidf_data_train.toarray())
        print(model_agglomerative_clustering.labels_)
        pca = PCA(n_components=2, random_state=21)
        reduced_features = pca.fit_transform(tfidf_data_train.toarray())

        # reduce the cluster centers to 2D
        reduced_cluster_centers = pca.transform(clf.cluster_centers_)
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clf.predict(tfidf_data_train))
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')
        plt.show()

        clustering = collections.defaultdict(list)
        for idx, label in enumerate(clf.labels_):
            clustering[label].append(idx)

        for key, value in clustering.items():
            print("Cluster #{0} Contains Rows --->".format(key), end='')
            for index in range(len(value)):
                print("[{0}] ".format(index), end='')
            print('\n')

    else:
        raise ValueError('Classifier model does not exist.')

    end_time = time.time()
    dispatchMessage()['sklearn_clustering'](False)
    print("Time to build initialize SKLearn Clustering Model:")
    print("Running Time: {0:.6f} Seconds.".format((end_time - start_time)))
    print("-----------------------------\n")


# KMeans
sklearn_clustering(loadObject(obj="pp_sklearn_dataframe.pl"), matrix_model="TFIDF", classifier_model="KMeans", k=14,
                   distance_mode='manhattan', linkage='single', metrics_='RAND')

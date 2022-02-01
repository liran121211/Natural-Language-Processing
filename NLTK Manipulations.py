import os
import pickle
import re
import time
import nltk
import numpy as np
import spacy
import textblob
import pandas as pd
import hunspell
import matplotlib.pyplot as plt

from spikex.pipes import AbbrX
from autocorrect import Speller
from wordcloud import WordCloud
from nltk.corpus import words, stopwords
from nltk import jaccard_distance, ngrams
from nltk.treeprettyprinter import TreePrettyPrinter


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
    saved_file = open(obj, 'rb')
    list_file = pickle.load(saved_file)
    saved_file.close()
    return list_file


def plotWordCloud(words_collection: str) -> None:
    cloud_1 = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(words_collection)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(cloud_1)
    plt.show()


def convertToLower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify DataFrame object so that every capital letter/word will be converted to lower case.
    :param df: pandas DataFrame object.
    :return: modified pandas DataFrame object.
    """

    dispatchMessage()['convertToLower'](True)
    for i, _ in df.iterrows():
        filtered_row = [re.sub('[^a-zA-Z]+', '', value.lower()) for value in df.at[i, 'Tweet Content'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'Tweet Content'] = filtered_row

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
        filtered_row = [re.sub('[^a-zA-Z]+', '', value) for value in df.at[i, 'Tweet Content'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'Tweet Content'] = filtered_row

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
        filtered_row = [value for value in df.at[i, 'Tweet Content'] if value not in stop_words]
        df.at[i, 'Tweet Content'] = filtered_row

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
        filtered_row = [re.sub('[^a-zA-Z]+', '', check_spell(value)) for value in df.at[i, 'Tweet Content'] if
                        len(re.sub('[^a-zA-Z]+', '', value)) > 0]
        df.at[i, 'Tweet Content'] = filtered_row

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


def nltk_spell_checker(tweet: str) -> list:
    """
    Spell checker using NLTK.
    :param tweet: String
    :return: list of corrected misspelled words.
    """
    tokenized_tweet = nltk_tokenize(tweet)
    nltk_words = words.words()
    corrected_words = []

    for word in tokenized_tweet:
        word_distances = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))), w) for w in nltk_words if
                          w[0] == word[0]]
        corrected_words += [(sorted(word_distances, key=lambda val: val[0])[0][1])]
    return corrected_words


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


def nltk_synonyms(tweet: str) -> list:
    """
    Return similar words using NLTK.
    :param tweet: String
    :return: list of synsets.
    """
    tokenized_tweet = nltk_tokenize(tweet)
    synsets = []
    try:
        for word in tokenized_tweet:
            for similar_words in nltk.corpus.wordnet.synsets(word):
                synsets += [similar_words.lemmas()[0].name()]
    except IndexError:
        synsets += [None]
    return list(set(synsets))


def nltk_ner(tweet: str) -> list:
    """
    Named Entity Recognition using NLTK.
    :param tweet: String
    :return: list of named entities.
    """
    tokenized_tweet = nltk_tokenize(tweet)
    nltk_pos_tagged = nltk.pos_tag(tokenized_tweet)
    nltk_ner_tagged = nltk.ne_chunk(nltk_pos_tagged, binary=False)
    return TreePrettyPrinter(nltk_ner_tagged).text()


def spacy_tokenize(text) -> list:
    """
    Tokenize text using Spacy.
    :param text: Tweet (String).
    :return: list of tokenized words.
    """
    spacy_nlp = spacy.load('en_core_web_lg')
    spacy_doc = spacy_nlp(text)
    return [token.text for token in spacy_doc]


def spacy_hunspell(tweet: str) -> list:
    """
    Spell checker using Spacy.
    :param tweet: String
    :return: list of corrected misspelled words.
    """
    spacy_nlp = spacy.load('en_core_web_lg')
    spacy_doc = spacy_nlp(tweet)
    spacy_words = [token.text for token in spacy_doc]
    spacy_spell_checker = hunspell.Hunspell('en_US')
    corrected_words = []

    for word in spacy_words:
        if spacy_spell_checker.spell(word):
            corrected_words += [word]
        else:
            corrected_words += [spacy_spell_checker.suggest(word)[0]]
    return corrected_words


def spacy_lemmatization(tweet: str) -> list:
    """
    Lemmatization using Spacy.
    :param tweet: String
    :return: list of lemmatized words.
    """
    sp = spacy.load('en_core_web_lg')
    spacy_tweet = sp(tweet)
    lemmatized_tweet = []

    for word in spacy_tweet:
        lemmatized_tweet += [word.lemma_]

    return lemmatized_tweet


def spacy_acronym(tweet: str) -> list:
    """
    Find Abbreviation using Spacy.
    :param tweet: String
    :return: list of abv words.
    """
    # This method does abbreviation only if the acronym is in the text along side.
    nlp = spacy.load("en_core_web_lg")

    nlp_tweet = nlp(tweet)
    spikex_abv = AbbrX(nlp.vocab)
    abv_tweet = spikex_abv(nlp_tweet)

    abv_tweet_list = []
    for abbr in abv_tweet._.abbrs:
        abv_tweet_list.append((abbr, abbr._.long_form))

    if len(abv_tweet_list) == 0:
        abv_tweet_list.append("No abbreviation has been found")
        return abv_tweet_list

    return abv_tweet_list


def spacy_synonyms(tweet: str, n_top_words=10) -> list:
    """
    Find similar words using Spacy.
    :param tweet: String
    :param n_top_words: max number of similar words per token (word).
    :return: list of similar words of all tokenized words.
    """
    nlp = spacy.load('en_core_web_lg')

    tokenized_tweet = spacy_tokenize(tweet)
    similar_list = []
    for word in tokenized_tweet:
        ms = nlp.vocab.vectors.most_similar(nlp(word).vector.reshape(1, nlp(word).vector.shape[0]), n=n_top_words)
        similar_list += [nlp.vocab.strings[w] for w in ms[0][0]]
    return similar_list


def spacy_ner(tweet: str) -> list:
    """
    Find Named Entities using Spacy.
    :param tweet: String
    :return: list of named entities.
    """
    nlp = spacy.load('en_core_web_lg')
    spacy_tweet = nlp(tweet)
    spacy_ner_tweet = []
    for word in spacy_tweet:
        ner = word.ent_type_
        spacy_ner_tweet += [ner if ner != '' else None]

    return spacy_ner_tweet


def textblob_tokenize(text) -> list:
    """
    Tokenize text using TextBlob.
    :param text: Tweet (String).
    :return: list of tokenized words.
    """
    return textblob.TextBlob(text).words


def textblob_spell_checker(tweet: str) -> list:
    """
    Spell checker using TextBlob.
    :param tweet: String
    :return: list of corrected misspelled words.
    """
    corrected_tweet = textblob.TextBlob(tweet).correct()
    return textblob_tokenize(str(corrected_tweet))


def textblob_lemmatization(tweet: str) -> list:
    """
    Lemmatization using TextBlob.
    :param tweet: String
    :return: list of lemmatized words.
    """

    # Uses NLTK library as extension for stemming!
    lemmatized_tweet = textblob.TextBlob(tweet).words
    return [textblob.Word(w).lemmatize() for w in lemmatized_tweet]


def textblob_stemming(tweet: str) -> list:
    """
    Stemming using TextBlob.
    :param tweet: String
    :return: list of stemmed words.
    """
    lemmatized_tweet = textblob.TextBlob(tweet).words
    return [textblob.Word(w).stem(nltk.PorterStemmer()) for w in lemmatized_tweet]


def textblob_synonyms(tweet: str) -> list:
    """
    Find similar words using TextBlob.
    :param tweet: String
    :return: list of similar words of all tokenized words.
    """
    tokenized_tweet = textblob_tokenize(tweet)
    similar_list = []
    for word in tokenized_tweet:
        ms = textblob.Word(word).synsets
        try:
            similar_list.append([textblob.Word(w).lemmatize() for w in ms[0].lemma_names()])
        except IndexError:
            similar_list += ["No synonyms has been found."]

    return similar_list


def textblob_wsd(tweet: str) -> list:
    """
    Find Word Sense Disambiguation using TextBlob.
    :param tweet: String
    :return: list of disambiguated words.
    """
    tokenized_tweet = textblob_tokenize(tweet)
    disambiguated_tweet = []
    for word in tokenized_tweet:
        disambiguated_tweet += [textblob.Word(word).definitions]
    return disambiguated_tweet


def create_dataframe() -> pd.DataFrame:
    """
    Create dataframe from local csv file.
    :return: pd.Dataframe.
    """
    dataframe = pd.read_csv('tweets.csv', delimiter=",")
    f_len = len(dataframe)

    for i, _ in dataframe.iterrows():
        print('\rConverting csv to dataframe: {0:.3f}%'.format(i / f_len * 100), end='')
        dataframe.at[i, 'Tweet Content'] = dataframe.at[i, 'Tweet Content'].split()
    print('All tweets tokenized...Done ✓\r', end='')

    return dataframe


def start() -> pd.DataFrame:
    pickle_file = os.getcwd() + "\\tweets.pl"
    pp_pickle_file = os.getcwd() + "\\pp_tweets.pl"
    print("\rChecking if Pickle file is already exists...", end='\n')

    if not os.path.exists(pickle_file):
        tweets_df = create_dataframe()
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

    return df_file


def nlp_manipulations(df: pd.DataFrame) -> dict:
    """
    Perform manipulations on Dataframe based on chosen function.
    :param df: pandas Dataframe.
    :return: manipulated dataframe.
    """

    def lemmatization(n_rows: int = 10) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        cloud_lemma_1 = ""
        cloud_lemma_2 = ""
        dispatchMessage()['lemmatization'](True)

        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'Tweet Content'])
            lemmatized_tweet = nltk_lemmatization(tweet)
            cloud_lemma_1 += tweet

            if lemmatized_tweet != list(df.at[i, 'Tweet Content']):
                df.at[i, 'Tweet Content'] = lemmatized_tweet
                cloud_lemma_2 += ' '.join(lemmatized_tweet)

        end_time = time.time()
        dispatchMessage()['lemmatization'](False)
        plotWordCloud(cloud_lemma_1)
        plotWordCloud(cloud_lemma_2)
        print("Running Time: {0:.6f} Seconds.".format(end_time - start_time))

    def stemming(n_rows: int = 10) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        cloud_lemma_1 = ""
        cloud_lemma_2 = ""
        dispatchMessage()['stemming'](True)

        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'Tweet Content'])
            cloud_lemma_1 += tweet
            stemmed_tweet = nltk_stemming(tweet)

            if stemmed_tweet != list(df.at[i, 'Tweet Content']):
                df.at[i, 'Tweet Content'] = stemmed_tweet
                cloud_lemma_2 += ' '.join(stemmed_tweet)

        end_time = time.time()
        dispatchMessage()['stemming'](False)
        plotWordCloud(cloud_lemma_1)
        plotWordCloud(cloud_lemma_2)
        print("Running Time: {0:.6f} Seconds.".format(end_time - start_time))

    def ner(n_rows: int = 10) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        ner_trees = []
        dispatchMessage()['ner'](True)
        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'Tweet Content'])
            ner_trees += [(tweet, nltk_ner(tweet))]
            # print("Tweet: [{0}]\n {1}\n".format(ner_trees[i][0],ner_trees[i][1]))

        end_time = time.time()
        dispatchMessage()['ner'](False)
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time) / 1000.0))

    def synonyms(n_rows: int = 10) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        word_synonyms = []
        tweet_synonyms = []

        dispatchMessage()['synonyms'](True)
        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = df.at[i, 'Tweet Content']
            str_tweet = ' '.join(tweet)
            word_synonyms.append([(w, textblob_synonyms(w)) for w in tweet])
            tweet_synonyms.append((str_tweet, word_synonyms[i]))

            print("\nTweet: ({0})".format(str_tweet))
            for w in word_synonyms[i]:
                print("Word: ({0}) Synonyms: {1}".format(w[0], w[1]))

        end_time = time.time()
        dispatchMessage()['synonyms'](False)
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time) / 1000.0))

    def acronym(n_rows: int = 10) -> None:
        """
        :param n_rows: number of rows to be manipulated.
        :return: None
        """
        start_time = time.time()
        acronym_list = []
        dispatchMessage()['acronym'](True)
        for i, _ in df.iloc[:n_rows].iterrows():
            tweet = ' '.join(df.at[i, 'Tweet Content'])
            acronym_list += [(tweet, spacy_acronym(tweet))]

        end_time = time.time()
        dispatchMessage()['acronym'](False)
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time) / 1000.0))

    return {
        'lemmatization': lemmatization,
        'stemming': stemming,
        'ner': ner,
        'synonyms': synonyms,
        'acronym': acronym
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
            print("Tokens are being stemmed...", end='')
        else:
            print("\rTokens are being stemmed...✓", end='\n')

    def ner(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being categorized as NER...", end='')
        else:
            print("\rTokens are being categorized as NER...✓", end='\n')

    def synonyms(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being searched for synonyms...", end='')
        else:
            print("\rTokens are being searched for synonyms...✓", end='\n')

    def acronym(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Tokens are being searched for acronyms...", end='')
        else:
            print("\rTokens are being searched for acronyms...✓", end='\n')

    def fill_matrix(mode: bool = True) -> None:
        """
        :param mode: True if function started, else false
        :return: None
        """
        if mode:
            print("Filling boolean matrix...", end='')
        else:
            print("\rFilling boolean matrix......✓", end='\n')

    return {
        'convertToLower': message_convertToLower,
        'removeSpecialChars': message_removeSpecialChars,
        'removeStopwords': message_removeStopwords,
        'fixSpellingError': message_fixSpellingError,
        'lemmatization': lemmatization,
        'stemming': stemming,
        'ner': ner,
        'synonyms': synonyms,
        'acronym': acronym,
        'fill_matrix': fill_matrix
    }


# start()


#####################################################################################

class BooleanMatrix:
    def __init__(self, data: pd.DataFrame, n_rows: int):
        self.data_ = data[:n_rows]
        self.c_size = None
        self.r_size = None
        self.words = None
        self.docs = None
        self.df = None

    def set_data(self) -> None:
        """
        Build dataframe for boolean matrix.
        :return: None
        """
        start_time = time.time()
        # Reload all words in tweets file.
        self.words = list(set([word for i, row in self.data_.iterrows() for word in row['Tweet Content']]))

        # Generate columns names for dataframe.
        self.docs = ['Doc #{0}'.format(i) for i in range(len(self.data_))]

        # set size for columns and rows.
        self.c_size = len(self.docs)
        self.r_size = len(self.words)

        # Initialize dataframe.
        self.df = pd.DataFrame(np.zeros((self.r_size, self.c_size)))

        # Set names for columns and rows.
        self.df.columns = self.docs
        self.df.index = self.words

        end_time = time.time()
        print("Time to build initialize object and fill it with empty data:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time) / 1000.0))

    def fill_matrix(self):
        dispatchMessage()['fill_matrix'](True)
        for i, row in self.data_.iterrows():
            for word in row['Tweet Content']:
                self.df.at[word, 'Doc #{0}'.format(i)] = 1
        dispatchMessage()['fill_matrix'](False)

    def query(self, q: str) -> pd.Series:
        """
        Search for documents who contains the specific keywords in the query.
        :param q: string of words and operators.
        :return: new boolean vector that represents the documents ID's relevant to the query.
        """
        def bitwise(vec1: pd.Series, vec2: pd.Series or None, op: str) -> pd.Series:
            """
            Perform bitwise operation between 2 boolean vectors (words).
            :param vec1: first word vector.
            :param vec2: second word vector.
            :param op: operator to be performed.
            :return: pd.Series after bitwise operation made.
            """
            bitwise_vec = []

            if op == 'AND':
                for idx in range(len(vec1)):
                    if vec1[idx] == 1 and vec2[idx] == 1:
                        bitwise_vec.append(1)
                    else:
                        bitwise_vec.append(0)

            if op == 'OR':
                for idx in range(len(vec1)):
                    if vec1[idx] == 0 and vec2[idx] == 0:
                        bitwise_vec.append(0)
                    else:
                        bitwise_vec.append(1)

            if op == 'NOT':
                for idx in range(len(vec1)):
                    if vec1[idx] == 1:
                        bitwise_vec.append(0)
                    else:
                        bitwise_vec.append(1)

            return pd.Series(bitwise_vec)

        # tokenized query to words.
        tokenized_query = q.split()

        # perform spell check for query.
        for i in range(len(tokenized_query)):
            if tokenized_query[i] == 'and' or tokenized_query[i] == '&':
                tokenized_query[i] = 'AND'

            if tokenized_query[i] == 'or' or tokenized_query[i] == '|':
                tokenized_query[i] = 'OR'

            if tokenized_query[i] == 'not' or tokenized_query[i] == '!':
                tokenized_query[i] = 'NOT'

        # split operators and words apart.
        operator_stack = [op for op in tokenized_query if (op == 'AND' or op == 'OR' or op == 'NOT')]
        tokens_stack = [t for t in tokenized_query if (t != 'AND' and t != 'OR' and t != 'NOT')]
        boolean_vectors = []
        temp = None

        # general check for bad query.
        if tokenized_query[0] == 'OR' or tokenized_query[0] == 'AND':
            print("ERROR: query cannot be started with merge (OR/NOT) operator.")
            exit(-1)

        # general check if only one word is given.
        if len(tokens_stack) == 1 and len(operator_stack) == 0:
            return pd.Series(['Doc #{0}'.format(i[0]) for i in self.df.loc[tokens_stack[0]].iteritems() if i[1] == 1])

        # create boolean vectors to all words.
        for i in range(len(tokens_stack)):
            try:
                boolean_vectors.append(self.df.loc[tokens_stack[i]])
            except KeyError as e:
                print("ERROR: [{0}] was not found in the structure.".format(e.args[0]))
                exit(-1)

        # keep first operator from stack.
        first_op = operator_stack.pop()

        if first_op == 'NOT':
            temp = bitwise(boolean_vectors.pop(), None, first_op)

        if first_op == 'AND' or first_op == 'OR':
            try:
                # check if next token needs to be complemented
                if operator_stack[-1:] == ['NOT']:
                    operator_stack.pop()

                    # complementation of the token.
                    temp = bitwise(boolean_vectors.pop(), None, 'NOT')

                    # merge it with the previous token.
                    temp = bitwise(temp, boolean_vectors.pop(), first_op)
                else:
                    # merge normally with OR or AND.
                    temp = bitwise(boolean_vectors.pop(), boolean_vectors.pop(), first_op)

            except KeyError as e:
                print("ERROR: [{0}] was not found in the structure.".format(e.args[0]))
                exit(-1)

        # check if there are more tokens in the stack.
        while len(operator_stack) > 0:
            # keep next operator from stack.
            op = operator_stack.pop()
            try:
                if op == 'NOT':
                    temp = bitwise(boolean_vectors.pop(), None, op)

                if op == 'AND' or op == 'OR':
                    # check if next token needs to be complemented
                    if operator_stack[-1:] == ['NOT']:
                        operator_stack.pop()
                        # complementation of the token.
                        temp = bitwise(boolean_vectors.pop(), None, 'NOT')
                    else:
                        # merge normally with OR or AND.
                        temp = bitwise(temp, boolean_vectors.pop(), op)
            except KeyError as e:
                print("ERROR: [{0}] was not found in the structure.".format(e.args[0]))
                exit(-1)

        return pd.Series(['Doc #{0}'.format(i[0]) for i in temp.iteritems() if i[1] == 1])


bm = BooleanMatrix(data=start(), n_rows=1000)
bm.set_data()
bm.fill_matrix()
# print(bm.query("nike"), '\n')
# print(bm.query("wonderful"), '\n')
# print(bm.query("not indeed"), '\n')
print(bm.query("nike or wonderful"), '\n')

# print(bm.query("nike"))
# print(bm.query("indeed"))


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

    def __repr__(self):
        if type(self.data) is str:
            return '[{0}]'.format(self.data)
        else:
            self.data.__repr__()


class LinkedList:
    def __init__(self):
        self.head = None

    def __repr__(self):
        pretty_print = ""
        if self.head is None:
            return "Empty Linked List..."

        while self.head is not None:
            if self.head.next is not None:
                pretty_print += "[{0}]---->".format(self.head.data)
            else:
                pretty_print += "[{0}]".format(self.head.data)
            self.head = self.head.next

        return pretty_print

    def insert(self, data: object, index: int = None, tail: bool = True) -> Node or None:
        """
        Insert data to linked list.
        :param data: object.
        :param index: specific position to insert data.
        :param tail: True to insert data to tail of linked list else at the beginning.
        :return: Node
        """

        if index is None:
            if tail is False:
                newNode = Node(data)
                newNode.next = self.head
                self.head = newNode
                return self.head

            else:
                new_node = Node(data)
                if self.head is None:
                    self.head = new_node

                    return new_node

                tail_node = self.head
                while tail_node.next:
                    tail_node = tail_node.next
                tail_node.next = new_node

                return new_node
        else:
            new_node = Node(data)

            if index < 1:
                return None

            elif index == 1:
                new_node.next = self.head
                self.head = new_node
            else:
                temp_node = self.head
                for i in range(1, index - 1):
                    if temp_node is not None:
                        temp_node = temp_node.next

                if temp_node is not None:
                    new_node.next = temp_node.next
                    temp_node.next = new_node
                else:
                    return None

    def pop(self) -> None or Node:
        """
        Pop last node from linked list.
        :return: Node
        """
        if self.head is not None:

            if self.head.next is None:
                self.head = None
                return self.head

            else:
                temp_node = self.head
                while temp_node.next.next is not None:
                    temp_node = temp_node.next

                tail_node = temp_node.next
                temp_node.next = None
                tail_node = None

                return temp_node

    def search(self, value: str, primitive: bool = True) -> Node or None:
        """
        Search for value in linked list.
        :param primitive: if Node value is primitive
        :param value: string.
        :return: Node.
        """
        if primitive is False:
            if self.head is None:
                return None

            # check if node is dictionary of linked list by itself, if so, then check if (value) already exists.
            if value in self.head.data.keys():
                return self.head

        else:
            temp_node = self.head
            i = 0

            if temp_node is not None:
                while temp_node is not None:
                    i += 1
                    if temp_node.data == value:
                        return temp_node
                    temp_node = temp_node.next
            else:
                return None

    def delete(self, index: int) -> Node or None:
        """
        Delete node from linked list.
        :param index: position of node to delete.
        :return: Node.
        """

        # if list is empty
        if self.head is None:
            return None

        # if list has only one node
        temp_node = self.head
        if index == 0:
            self.head = temp_node.next
            temp_node = None
            return self.head

        # traverse list till position
        for i in range(index - 1):
            temp_node = temp_node.next
            if temp_node is None:
                break

        # If position is more than number of nodes
        if temp_node is None:
            return
        if temp_node.next is None:
            return

        # Now temp.next is the node to be deleted
        next_node = temp_node.next.next

        # Unlink the node from linked list
        temp_node.next = None

        temp_node.next = next_node

    def size(self):
        """
        Return the size of the linked list.
        :return: Node.
        """
        temp_node = self.head
        i = 0
        if temp_node is not None:
            while temp_node is not None:
                i += 1
                temp_node = temp_node.next
        else:
            return 0
        return i


class ReversedIndexes:
    def __init__(self, data: pd.DataFrame, n_rows: int):
        self.structure = dict()
        self.data_ = data[:n_rows]
        self.r_size = None
        self.words = None

    def set_data(self) -> None:
        """
        Build dataframe for boolean matrix.
        :return: None
        """
        start_time = time.time()

        # Reload all words in tweets file.
        self.words = list(set([word for i, row in self.data_.iterrows() for word in row['Tweet Content']]))

        # Initialize linkedlist structure for every word.
        for w in self.words:
            self.structure[w] = LinkedList()

        for i, row in self.data_.iterrows():
            for w in row['Tweet Content']:
                # insert only unique values.
                if self.structure[w].search(value=i, primitive=True) is None:
                    self.structure[w].insert(i)

        end_time = time.time()
        print("Time to build linked list:")
        print("Running Time: {0:.6f} Seconds.".format((end_time - start_time) / 1000))

    def print(self) -> None:
        for key, value in self.structure.items():
            print('[Size: {0}] {1} ===> '.format(value.size(), key), end='')
            print(self.structure[key])

    def merge(self, sp_interval: int = 1) -> dict:
        """
        Merge 2 linked lists under clousers of intersection, union and implementation.
        :param sp_interval: movement steps between each node in the linked list.
        :return: new linked list.
        """

        def skip_interval(posting1: LinkedList) -> LinkedList:
            for i in range(sp_interval):
                if posting1.head is not None:
                    posting1.head = posting1.head.next
            return posting1

        def intersection(posting1: LinkedList, posting2: LinkedList) -> LinkedList:
            """
            Intersection of 2 linked lists.
            :param posting1: LinkedList.
            :param posting2: LinkedList.
            :return: new linked list.
            """
            answer = LinkedList()
            while posting1.head is not None and posting2.head is not None:
                if posting1.head.data == posting2.head.data:
                    answer.insert(posting1.head.data)
                    skip_interval(posting1)
                    skip_interval(posting2)
                else:
                    if posting1.head.data < posting2.head.data:
                        skip_interval(posting1)
                        skip_interval(posting2)

                skip_interval(posting1)
                skip_interval(posting2)

            return answer

        def union(posting1: LinkedList, posting2: LinkedList) -> LinkedList:
            """
            Union of 2 linked lists.
            :param posting1: LinkedList.
            :param posting2: LinkedList.
            :return: new linked list.
            """
            answer = LinkedList()

            while posting1.head is not None:
                answer.insert(posting1.head.data)
                skip_interval(posting1)

            while posting2.head is not None:
                # Intersect duplicate values that already appears in answer.
                if answer.search(value=posting2.head.data, primitive=True) is None:
                    answer.insert(posting2.head.data)
                skip_interval(posting2)

            return answer

        def complement(word: str) -> LinkedList:
            """
            complementation of linked list.
            :param word: word (String).
            :return: new linked list.
            """
            answer = LinkedList()
            # Iterate over all linked lists in the structure.
            for key, value in self.structure.items():
                if word != key:
                    # Iterate over all words in the specific linked list.
                    if value.head is not None:
                        if answer.search(value=value.head.data, primitive=True) is None:
                            answer.insert(value.head.data)
            return answer

        return {
            'AND': intersection,
            'OR': union,
            'NOT': complement
        }

    def query(self, q: str, skip_inv: int = 1) -> LinkedList:
        """
        Get the documents that contains the words in the query.
        :param q: words and boolean operators.
        :param skip_inv: interval of skipping between each node in the linked lists.
        :return: new linked list with the contains nodes with the id's of the documents.
        """
        tokenized_query = q.split()
        tokenized_query.reverse()

        # perform spell check for query.
        for i in range(len(tokenized_query)):
            if tokenized_query[i] == 'and' or tokenized_query[i] == '&':
                tokenized_query[i] = 'AND'

            if tokenized_query[i] == 'or' or tokenized_query[i] == '|':
                tokenized_query[i] = 'OR'

            if tokenized_query[i] == 'not' or tokenized_query[i] == '!':
                tokenized_query[i] = 'NOT'

        temp = None

        # split operators and words apart.
        operator_stack = [op for op in tokenized_query if (op == 'AND' or op == 'OR' or op == 'NOT')]
        tokens_stack = [t for t in tokenized_query if (t != 'AND' and t != 'OR' and t != 'NOT')]

        if tokenized_query[0] == 'OR' or tokenized_query[0] == 'NOT':
            print("ERROR: query cannot be started with merge (OR/NOT) operator.")
            exit(-1)

        # general check if only one word is given.
        if len(tokens_stack) == 1 and len(operator_stack) == 0:
            return self.structure[tokens_stack[0]]

        # keep first operator from stack.
        first_op = operator_stack.pop()

        if first_op == 'NOT':
            temp = self.merge(skip_inv)[first_op](tokens_stack.pop())

        if first_op == 'AND' or first_op == 'OR':
            # keep first token from stack.
            first_token = tokens_stack.pop()
            try:
                # check if next token needs to be complemented
                if operator_stack[-1:] == ['NOT']:
                    operator_stack.pop()

                    # complementation of the token.
                    temp = self.merge(skip_inv)['NOT'](tokens_stack.pop())

                    # merge it with the previous token.
                    temp = self.merge(skip_inv)[first_op](temp, self.structure[first_token])
                else:

                    # merge normally with OR or AND.
                    temp = self.merge(skip_inv)[first_op](self.structure[first_token],
                                                          self.structure[tokens_stack.pop()])
            except KeyError as e:
                print("ERROR: [{0}] was not found in the structure.".format(e.args[0]))
                exit(-1)

        # check if there are more tokens in the stack.
        while len(operator_stack) > 0:
            # keep next operator from stack.
            op = operator_stack.pop()
            try:
                if op == 'NOT':
                    temp = self.merge(skip_inv)[op](self.structure[tokens_stack.pop()])

                if op == 'AND' or op == 'OR':

                    # check if next token needs to be complemented
                    if operator_stack[-1:] == ['NOT']:
                        operator_stack.pop()
                        # complementation of the token.
                        next_temp = self.merge(skip_inv)['NOT'](tokens_stack.pop())

                        # merge it with the previous token.
                        temp = self.merge(skip_inv)[op](temp, next_temp)
                    else:
                        # merge normally with OR or AND.
                        temp = self.merge(skip_inv)[op](temp, self.structure[tokens_stack.pop()])
            except KeyError as e:
                print("ERROR: [{0}] was not found in the structure.".format(e.args[0]))
                exit(-1)

        return temp

class ReversedIndexesImproved(ReversedIndexes):
    def __init__(self, data: pd.DataFrame, n_rows: int):
        super(ReversedIndexes, self).__init__()
        self.structure = dict()
        self.data_ = data[:n_rows]

    def set_data(self) -> None:
        """
        Build dataframe for boolean matrix.
        :return: None
        """
        # Reload all words in tweets file.
        self.words = list(set([word for i, row in self.data_.iterrows() for word in row['Tweet Content']]))

        # Initialize linkedlist structure for every word.
        for w in self.words:
            self.structure[w] = LinkedList()

        for i, row in self.data_.iterrows():
            for w in row['Tweet Content']:
                tweet = ' '.join(row['Tweet Content'])
                word_indexes = LinkedList()

                # insert only unique values
                # if self.structure[w].search(value=i, primitive=False) is None:
                _ = [word_indexes.insert(w.start()) for w in re.finditer(w, tweet)]
                self.structure[w].insert({i: word_indexes})

    def print(self) -> None:
        for word, main_list in self.structure.items():
            print('Word: [{0}]'.format(word), end='\n')
            while main_list.head is not None:
                for doc_id, sub_list in main_list.head.data.items():
                    print("     [Doc #{0}]".format(doc_id), end='\n')
                    print('         [Indexes:', end='')
                    while sub_list.head is not None:
                        print(" #{0}".format(sub_list.head.data), end='')
                        sub_list.head = sub_list.head.next
                print(']', end='\n\n')
                main_list.head = main_list.head.next


rev_list = ReversedIndexes(data=start(), n_rows=1000)
rev_list.set_data()
print(rev_list.query(q="not indeed or wonderful", skip_inv=1))
# rev_list.query(q="nike", skip_inv=1).pretty_print()
# rev_list.query(q="indeed", skip_inv=1).pretty_print()
# print(rev_list.query(q="nike or wonderful", skip_inv=1).pretty_print())

# rev_list_improved = ReversedIndexesImproved(data=start(), n_rows=1000)
# rev_list_improved.set_data()
# print(rev_list_improved.query(q="wonderful OR indeed", skip_inv=1))

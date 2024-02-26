import math
import re
from collections import Counter
from os.path import isfile

import joblib
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords, wordnet
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from unidecode import unidecode

DATA_REPOSITORY = 'analysis/data/'
MODEL_REPOSITORY = 'analysis/model/'
LANGUAGE = 'french'
LANGUAGE_CODE = 'fra'
STOP_WORDS = set(stopwords.words(LANGUAGE))

# warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def get_original_df(limit, csv_name='new.csv',
                    encoding='utf-8',
                    header=None):
    file_name = f'{DATA_REPOSITORY}email.original.{limit}.pq'
    if isfile(file_name):
        email_df = pd.read_parquet(file_name)
    else:
        email_df = pd.read_csv(f'{DATA_REPOSITORY}{csv_name}', encoding=encoding, header=header)
        email_df.to_parquet(file_name)
    if limit:
        limit = limit // 2
        return pd.concat([email_df[:limit], email_df[-limit:]], ignore_index=True, axis=0)
    return email_df


def show(text, log):
    if log:
        print(text)


def get_synonym(word):
    synonyms = []
    for syn in wordnet.synsets(word, lang='fra'):
        for lemma in syn.lemmas(lang='fra'):
            synonyms.append(lemma.name())
    return list(set(synonyms))


def get_tokens(text, stop_words=None, log=False):
    if stop_words is None:
        stop_words = STOP_WORDS

    text = unidecode(text).lower()  # utf8
    show(text, log)

    soup = BeautifulSoup(text, 'html.parser')  # html
    text = soup.get_text()
    show(text, log)

    text = re.sub(r'http[s]?://\S+', ' ', text)  # link
    text = re.sub(r'@\w+', ' ', text)  # mention
    text = re.sub(r'[^\w\s]', ' ', text)  # punctuations
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)  # numbers
    show(text, log)

    tokens = nltk.word_tokenize(text, language=LANGUAGE)
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language=LANGUAGE)
    # stemmer = PorterStemmer()

    n_tokens = []
    for token in tokens:
        if token not in stop_words:
            n_tokens.append(stemmer.stem(token))
    show(n_tokens, log)

    return n_tokens


def get_tokens_with_frequency(text, stop_words=None, log=False):
    return Counter(get_tokens(text, stop_words, log))


def get_frequency_df(limit, text_column):
    file_name = f'{DATA_REPOSITORY}email.frequency.{limit}.pq'
    if isfile(file_name):
        freq_df = pd.read_parquet(file_name)
    else:
        email_df = get_original_df(limit)
        tokens = dict(email_df[text_column].apply(get_tokens_with_frequency))
        tdf = pd.DataFrame.from_dict(tokens, orient="index")
        freq_df = pd.concat([email_df, tdf], axis=1).fillna(0)
        freq_df.to_parquet(file_name)
    return freq_df


"""def get_document_frequency(column):
    return column.shape[0] - column.value_counts().get(0, 0)


def get_words_documents_frequency(frequency_df=None, limit=None, start_column=6):
    if frequency_df is None:
        print("Lasa anaty memoire indroa ny frequency_df")
        frequency_df = get_frequency_df(limit)
    return frequency_df.iloc[:, start_column:].apply(get_document_frequency, axis=0)"""


def get_document_inverse_frequency(column):
    return math.log(column.shape[0] / (column.shape[0] - column.value_counts().get(0, 0)))


def get_words_inverse_documents_frequency(frequency_df=None, limit=None, start_column=2):
    if frequency_df is None:
        frequency_df = get_frequency_df(limit, 1)
    return frequency_df.iloc[:, start_column:].apply(get_document_inverse_frequency, axis=0)


def get_tf_idf_df(limit, start_column=2, text_column=1):
    file_name = f'{DATA_REPOSITORY}email.tf_idf.{limit}.pq'
    if isfile(file_name):
        tf_idf_df = pd.read_parquet(file_name)
    else:
        tf_idf_df = get_frequency_df(limit, text_column=text_column)
        idf_values = get_words_inverse_documents_frequency(tf_idf_df)
        tf_idf_df.iloc[:, start_column:] = tf_idf_df.iloc[:, start_column:].mul(idf_values)
        tf_idf_df.to_parquet(file_name)
    return tf_idf_df


def get_df_important_column(limit, start_column, column_limit):
    file_name = f'{DATA_REPOSITORY}email.important.{limit}.{column_limit}.pq'
    if isfile(file_name):
        df = pd.read_parquet(file_name)
    else:
        dataframe = get_tf_idf_df(limit, start_column=start_column)
        column_sums = dataframe.iloc[:, start_column:].sum()
        sorted_columns = column_sums.sort_values(ascending=False)
        top_columns = sorted_columns[:column_limit]
        df = pd.concat([dataframe.iloc[:, :start_column], dataframe[top_columns.index]], axis=1)
    return df


def get_model_with_words_idf(data_limit, model, column_limit, x_column_start=2, y_column=0, test_size=0.2,
                             score=False, **model_args):
    file_name = f'{MODEL_REPOSITORY}{data_limit}.{model_args}.{model}'
    words_idf_file_name = f'{MODEL_REPOSITORY}{data_limit}.{model_args}.{model}.words_idf'

    if isfile(file_name) and isfile(words_idf_file_name):
        model = joblib.load(file_name)
        words_idf = joblib.load(words_idf_file_name)
    else:
        df = get_df_important_column(data_limit, x_column_start, column_limit)
        all_words_idf = get_words_inverse_documents_frequency(get_frequency_df(data_limit, 1))
        words_idf = {}
        for c in df.columns[x_column_start:]:
            words_idf[c] = all_words_idf.loc[c]
        words_idf = pd.DataFrame.from_dict(words_idf, orient='index')

        x = df.iloc[:, x_column_start:x_column_start+column_limit]
        y = df.iloc[:, y_column]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

        model = model(**model_args).fit(x_train, y=y_train)
        joblib.dump(model, file_name)
        joblib.dump(words_idf, words_idf_file_name)

        if score:
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_test, y_pred))
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

    return model, words_idf


def prediction(model, words_idf, text, log=True):
    tokens = get_tokens_with_frequency(text, log=log)
    to_predict = {words_idf.index.values[0]: 0}
    for token in tokens.keys():
        if token in words_idf.index.values:
            to_predict[token] = tokens.get(token)*words_idf.loc[token].values[0]
        else:
            for syn in get_synonym(token):
                if syn in words_idf.keys():
                    to_predict[token] = tokens.get(token) * words_idf.get(syn)
    tf_idf_df = pd.DataFrame.from_dict({0 : to_predict}, columns=words_idf.index.values, orient='index').fillna(0)
    return model.predict(tf_idf_df)

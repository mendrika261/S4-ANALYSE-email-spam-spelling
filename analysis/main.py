import numpy as np
from sklearn.svm import SVC

from analysis.classification import *


def get_df_important_column(limit, start_column, column_limit):
    dataframe = get_tf_idf_df(limit, start_column=start_column)
    column_sums = dataframe.iloc[:, start_column:].sum()
    sorted_columns = column_sums.sort_values(ascending=False)
    top_columns = sorted_columns[:column_limit]
    return pd.concat([dataframe.iloc[:, :start_column], dataframe[top_columns.index]], axis=1)


if __name__ == '__main__':
    # nltk.download('popular')
    # df = get_original_df(100, csv_name='chat.csv', encoding='utf-8', header=None)
    # df = get_tf_idf_df(100, start_column=2, text_column=1)
    # df = get_df_important_column(100, 2, 500)

    model, words_idf = get_model_with_words_idf(data_limit=100, model=SVC, kernel='linear', column_limit=500,
                                                x_column_start=2, y_column=0, test_size=0.2, score=False)

    print(prediction(model, words_idf, """Objet: Gagner de l'argent rapidement et facilement ! Contenu: Bonjour, 
    cher ami, avez-vous besoin d'argent rapidement et facilement ? Nous avons la solution pour vous ! Notre système 
    révolutionnaire vous permet de générer des revenus illimités sans effort. Il vous suffit de suivre nos 
    instructions simples et vous serez sur la voie de la richesse en un rien de temps. Ne manquez pas cette 
    opportunité incroyable. Rejoignez-nous dès maintenant et commencez à gagner de l'argent dès aujourd'hui !""",
                     log=True))

    # train classification
    """x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 2:], df.iloc[:, 0], test_size=0.2, random_state=0)

    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import nltk

def feature_engineer(x):
    x = handle_name(x)
    x = handle_description(x)
    return x


def handle_name(x):
    """
    :param x:
    :return: X with added two features and removed Name feature:
    1. name_type: categorical
    values:
        0: no name 'nan'
        1: multiple names
        2: not a real name
        3: real name
    2. name length: scalar.
    """
    names = x.Name
    name_type = list(map(lambda x: 'NO_NAME' if type(x) == float else x, names))

    def cond(a):
        if a == 'NO_NAME':  # no name (nan)
            return 0
        elif '&' in a or 'and' in a.lower() or ',' in a:  # multiple names
            return 1
        elif ' ' in a:  # not actual name
            return 2
        else:  # real name
            return 3
    name_type = list(map(cond, name_type))
    name_length = list(map(lambda x: 0 if type(x) == float else len(x), names))
    x = x.drop('Name', axis=1)
    x['name_type'] = name_type
    x['name_length'] = name_length
    return x


def handle_description(x):
    description = x['Description']
    description = description.fillna("none")
    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Fit TFIDF
    tfv.fit(list(description))
    description = tfv.transform(description)
    print("description (tfidf):", description.shape)
    svd = TruncatedSVD(n_components=120)
    svd.fit(description)
    # print(svd.explained_variance_ratio_.sum())
    # print(svd.explained_variance_ratio_)
    description = svd.transform(description)
    print("description (svd):", description.shape)
    new_cols = pd.DataFrame(description, columns=['description_svd_{}'.format(i) for i in range(120)])
    x = pd.concat((x, new_cols), axis=1)
    x = x.drop('Description', axis=1)
    return x

















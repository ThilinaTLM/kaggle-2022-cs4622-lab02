import pandas as pd
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt


def normalize(df, columns):
    for column in columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


def draw_normal_distribution(df, columns):
    df = df[columns]
    df.hist(bins=50, figsize=(10, 6))
    plt.show()


def feature_selection_corr(X, y, threshold=0.5):
    df = pd.concat([X, y], axis=1)
    corr = df.corr()

    # get correlation with target
    corr_with_target = corr[y.name]

    # sort by absolute value
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)

    return corr_with_target[corr_with_target > threshold].index


def check_feature_corr(X, threshold=0.5):
    corr = X.corr()
    columns = corr.columns

    # create a dataframe all column pairs
    df = pd.DataFrame(columns=['feature1', 'feature2', 'corr'])
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            df = pd.concat(
                [df,
                 pd.DataFrame(
                     [[columns[i], columns[j], corr[columns[i]][columns[j]]]],
                     columns=['feature1', 'feature2', 'corr']
                 )],
            )

    # sort by absolute value
    df['corr'] = df['corr'].abs()
    df = df.sort_values(by='corr', ascending=False)

    # filter by threshold
    df = df[df['corr'] > threshold]

    return df


def feature_selection_chi2(X, y, threshold=0.5):
    chi2_score, p_value = chi2(X, y)
    df = pd.DataFrame({'chi2': chi2_score, 'p_value': p_value}, index=X.columns)
    return df[df['p_value'] < threshold].index

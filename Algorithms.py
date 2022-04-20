import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2





trainPath = 'class_train.csv'
testPath = 'class_test.csv'


def split_dataset(df=None, filepath=None):
    if filepath:
        df = pd.read_csv(filepath).dropna()
    outcome = df.loc[:, ['Outcome']].values

    df.drop('Outcome', axis=1, inplace=True)
    features = df.columns
    features = df.loc[:, features].values

    return features, outcome.ravel()


def naive_bayes(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues):
    gnb = GaussianNB()
    gnb.fit(trainingFeatures, trainingOutcome)
    prediction = gnb.predict(testFeatures)
    print_scores(testLabels, prediction, "Naive Bayes", betaValues)


def decision_tree(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues):
    dtc = DecisionTreeClassifier()
    fig = dtc.fit(trainingFeatures, trainingOutcome)
    prediction = dtc.predict(testFeatures)
    print_scores(testLabels, prediction, "Decision Tree", betaValues)
    # plot_tree(fig)
    # plt.show()


def support_vector_machine(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues):
    svc = svm.SVC()
    svc.fit(trainingFeatures, trainingOutcome)
    prediction = svc.predict(testFeatures)
    print_scores(testLabels, prediction, "Support Vector Machine", betaValues)


def k_neighbors(trainingFeatures, trainingOutcome, testFeatures, testLabels, n_neighbors, betaValues):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(trainingFeatures, trainingOutcome)
    prediction = neigh.predict(testFeatures)
    print_scores(testLabels, prediction, "K-Nearest Neighbor", betaValues)


def print_scores(testLabels, prediction, modelName, betaValues):
    print(f"\n{modelName}:")
    print(f"Accuracy score: {accuracy_score(testLabels, prediction)}")
    for value in betaValues:
        print(f"f({value})-score: {fbeta_score(testLabels, prediction, average='weighted', beta=value)}")


def combine_csv_files(filePathList):
    df = pd.read_csv(filePathList[0])
    # df2 = pd.read_csv(filePathList[1])
    # print(df.head())
    # print('####################################################################')
    # print(df2.head())

    for path in filePathList[1:]:
        # df_merge = df.merge(pd.read_csv(path), how='left').dropna()
        dfConcat = pd.concat([df, pd.read_csv(path)], ignore_index=True)

    # dfConcatNoIndex = dfConcat.drop('index', inplace=True)
    print(dfConcat['index'].head())
    # print(dfConcatNoIndex.head())


def extract_minority_class(dataFrame, testSize=0.2):
    dfClassZero = dataFrame[dataFrame.Outcome == 0]
    test = dfClassZero.sample(frac=testSize)
    train = dataFrame.drop(test.index)
    return train, test


def features_selection(df_in):
    X = df_in.iloc[:, 0:159].values
    y = df_in.iloc[:, 159:160].values

    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, y)

    np.set_printoptions(precision=3)
    features = fit.transform(X)

    features = features[0]
    num_to_check = []
    for i in features:
        num = i
        num = round(num, 3)
        num_to_check.append(num)

    find_column = df_in.iloc[0]
    find_column = find_column.values

    values = []
    for i in find_column:
        num = i
        num = round(num, 3)
        values.append(num)

    index_list = []
    for i in num_to_check:
        num = i
        index = values.index(num)
        index_list.append(index)

    df_out = df_in.iloc[:, index_list]
    return df_out


def run():

    trainingFeatures, trainingOutcome = split_dataset(filepath=trainPath)
    testFeatures, testLabels = split_dataset(filepath=testPath)

    betaValues = [1]
    naive_bayes(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues)
    decision_tree(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues)
    support_vector_machine(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues)
    k_neighbors(trainingFeatures, trainingOutcome, testFeatures, testLabels, 3, betaValues)


    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)
    mergedDataFrame = pd.concat([pd.read_csv(trainPath), pd.read_csv(testPath)], ignore_index=True)
    fsDataFrame = features_selection(mergedDataFrame)
    mergedOutcome = mergedDataFrame.loc[:, ['Outcome']].values
    fsDataFrame['Outcome'] = mergedOutcome
    train, test = extract_minority_class(fsDataFrame, 0.2)
    trainingFeatures, trainingOutcome = split_dataset(df=train)
    testFeatures, testLabels = split_dataset(df=test)
    betaValues = [0.5, 1, 2]
    naive_bayes(trainingFeatures, trainingOutcome, testFeatures, testLabels, betaValues)
    k_neighbors(trainingFeatures, trainingOutcome, testFeatures, testLabels, 3, betaValues)


if __name__ == "__main__":
    run()
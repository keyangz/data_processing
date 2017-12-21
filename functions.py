import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error, recall_score, precision_score, f1_score, \
    accuracy_score, auc

import missingno as msno
import warnings

imp = Imputer(missing_values = np.NaN, strategy='median', axis=0)
ss = StandardScaler()

def data_type_plot(mydata):
    dataTypeDf = pd.DataFrame(mydata.dtypes.value_counts()).reset_index().rename(
        columns={"index": "variableType", 0: "count"})
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    sns.barplot(data=dataTypeDf, x="variableType", y="count", ax=ax, color="#34495e")
    ax.set(xlabel='Variable Type', ylabel='Count', title="Variables Count Across Datatype")

def missing_value_plot(mydata):
    missingValueColumns = mydata.columns[mydata.isnull().any()].tolist()
    msno.bar(mydata[missingValueColumns], figsize=(20, 8), color="#34495e", fontsize=12, labels=True)

def split_data(mydata, target):
    """
    Split raw data to train and test
    :param mydata: raw data
    :param target: response variable
    :return: X_train, X_test, y_train, y_test
    """
    y = mydata[target]
    X = mydata.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    return X_train, X_test, y_train, y_test

def get_eda_data(X_train, y_train, target):
    """
    :param mydata: data frame
    :param target: response variable
    :param y_train: y_train
    :return: numerical and categorical features with y, ready for EDA
    """
    numericData = X_train.select_dtypes(include=[np.number])
    categoryData = X_train.select_dtypes(include=[object])
    numericData = pd.concat([numericData, y_train], axis=1)
    categoryData = pd.concat([categoryData, y_train], axis=1)

    return numericData, categoryData

def distribution_vs_class(numericData, target):
    """
    CLASSIFICATION PROBLEM (response vs numerical) Also consider violin plot in seaborn
    :param numericData: numerical features
    :param target: response variable
    :return: make distribution plots of numerical features for each class
    """
    for i, col in enumerate(numericData.columns):
        if col != target:
            facet = sns.FacetGrid(numericData, hue="Survived", aspect=4)
            facet.map(sns.kdeplot, col, shade=True)
            facet.set(xlim=(0, numericData[col].max()))
            facet.add_legend()

def count_vs_class(categoryData, target):
    """
    CLASSIFICATION PROBLEM (response vs categorical) Also consider factor plot in seaborn
    :param categoryData: categorical features
    :param target: response variable
    :return: count plot in seaborn to show the count of each class for each categorical variable
    """
    for i, col in enumerate(categoryData.columns):
        if col != target:
            plt.figure(i)
            sns.countplot(x=target, hue=col, data=categoryData)

def scatterPlot(numericData, target):
    """
    REGRESSION PROBLEM (response vs numerical)
    :param numericData: numerical features
    :param target: response variable
    :return: scatter plots of response vs features
    """
    for i, col in enumerate(numericData.columns):
        if col != target:
            plt.figure(i)
            sns.regplot(x = col, y = target, data=numericData)

def boxPlot(categoryData, target):
    """
    REGRESSION PROBLEM (response vs categorical)
    :param categoryData: categorical features
    :param target: response variable
    :return: boxplots of features
    """
    for i, col in enumerate(categoryData.columns):
        if col != target:
            plt.figure(i)
            sns.boxplot(x = col, y = target, data = categoryData)

def getCorrDF(numericData, target):
    """
    REGRESSION PROBLEM: GET CORRELATION DF for target and numerical features
    :param numericData: numerical features
    :param target: response variable
    :return: correlation data frame
    """
    corr_df = numericData.corr().abs()
    ss = corr_df.unstack()
    corr_df = pd.DataFrame(ss[target].sort_values(ascending = False)[1:]).reset_index()
    corr_df.columns = ['feature', 'correlation']

    return corr_df

def correlationHeatmap(numericData):
    """
    GET CORRELATION HEATMAP
    :param numericData: numerical features and response variable
    :return: heatmap
    """
    f, ax = plt.subplots(figsize=(11, 9))
    plt.title('Pearson Correlation Matrix')
    sns.heatmap(numericData.corr(),linewidths=0.5,vmax=0.3,square=True, cmap="YlGnBu",center=0, linecolor='black',
                annot=True, cbar_kws={"shrink": .5})


def get_num_cat(X_train):
    """
    Split training X to numerical and categorical features, then deal with each features separately
    :param X_train: X_train
    :return: numerical and categorical features
    """
    numericTrain = X_train.select_dtypes(include=[np.number])
    categoryTrain = X_train.select_dtypes(include=[object])

    return numericTrain, categoryTrain


def numeric_train_transform(numericTrain):
    """
    Filling NA with median, then standardizing
    :param numericTrain: numerical features
    :return: transformed numerical features
    """
    numericTrain = numericTrain.fillna(np.NaN)
    numericAfter = pd.DataFrame(ss.fit_transform(imp.fit_transform(numericTrain)), columns=numericTrain.columns)

    return numericAfter


def category_train_transform(categoryTrain):
    """
    Filling NA with Unknown
    :param categoryTrain: categorical features
    :return: transformed categorical features
    """
    categoryTrain = categoryTrain.fillna('Unknown')
    categoryAfter = pd.get_dummies(categoryTrain)
    categoryAfter = categoryAfter.reset_index()
    categoryAfter.drop(['index'], axis=1, inplace=True)

    return categoryAfter


def combine_num_cat(numericAfter, categoryAfter):
    """
    Combine the transformed numerical and categorical features
    :param numericAfter: transformed numerical features
    :param categoryAfter: transformed categorical features
    :return: combined data frames
    """
    return pd.concat([numericAfter, categoryAfter], axis=1)


def y_reset_index(y):
    """y
    Reset index for y
    :param y: response variable
    :return: y after resetting index
    """
    y = pd.DataFrame(y).reset_index().iloc[:, 1]

    return y

def xgboost_feature_importance(combinedTrain, y_train, objective, score):
    """
    XGBoost feature importance
    :param combinedTrain: training data
    :param y_train: response
    :param objective: regression or classification
    :param score: metric to look at
    :return: plot of feature importance
    """
    ## https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    ## objective: "reg:linear", "reg:logistic"
    xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': objective,
    'eval_metric': score,
    'silent': 1
    }
    dtrain = xgb.DMatrix(combinedTrain, y_train, feature_names = combinedTrain.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
    featureImportance = model.get_fscore()
    features = pd.DataFrame()
    features['features'] = featureImportance.keys()
    features['importance'] = featureImportance.values()
    features.sort_values(by=['importance'],ascending=False,inplace=True)
    fig,ax= plt.subplots()
    fig.set_size_inches(20,10)
    plt.xticks(rotation=90)
    sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")

def extra_tree_importance(combinedTrain, y_train, objective):
    """
    Extra trees feature importance
    :param combinedTrain: training data
    :param y_train: response variable
    :objective: regression or classification
    :return: plot of extra trees feature importance
    """
    feat_names = combinedTrain.columns.values
    if objective == 'regression':
        model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=5, max_features=0.7, n_jobs=-1, random_state=0)
    else:
        model = ensemble.ExtraTreesClassifier(n_estimators=25, max_depth=5, max_features=0.7, n_jobs=-1, random_state=0)
    model.fit(combinedTrain, y_train)

    ## plot the importances ##
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()

def compare_cv_classifiers(combinedTrain, y_train, scoring = 'accuracy', verbose = 2):
    """
    Comparing performances of classifiers on cross validation
    :param combinedTrain: training data
    :param y_train: response variable
    :param scoring: metro
    :verbose: 1: plot, 2: print numbers
    :return: bar plot of scores
    """
    classifiers = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier(),
              SVC(kernel="rbf", C=0.025), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
               KNeighborsClassifier(), ExtraTreesClassifier()]
    classifier_names = ["Logistic", "RandomForest", "GradientBoost", "XGBoost", "SVM", "LDA", "QDA", "KNN",
                        "ExtraTrees"]
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    cols = ['Classifiers', 'Validation_Score']
    compareDF = pd.DataFrame(columns = cols)
    for i in xrange(len(classifiers)):
        cv_results = cross_val_score(classifiers[i], combinedTrain, y_train, cv=kfold, scoring='accuracy')
        compareDF.loc[i] = [classifier_names[i], np.mean(cv_results)]
    if verbose == 1:
        sns.barplot(x="Classifiers", y="Validation_Score", data=compareDF)
    elif verbose == 2:
        print compareDF

def compare_cv_regressors(combinedTrain, y_train, scoring = 'neg_mean_squared_error', verbose = 2):
    """
    Comparing performances of classifiers on cross validation
    :param combinedTrain: training data
    :param y_train: response variable
    :param scoring: metro
    :return: bar plot of scores
    """
    regressors = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor(),
               Lasso(alpha = 0.1), Ridge(alpha=0.2), ExtraTreesRegressor(), ElasticNet(), KNeighborsRegressor()]
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    regressor_names = ["Linear", "RandomForest", "GradientBoost", "XGBoost", "Lasso", "Ridge", "ExtraTrees",
                        "ElasticNet", "KNN"]
    cols = ['Regressors', 'Validation_Score']
    compareDF = pd.DataFrame(columns = cols)
    for i in xrange(len(regressors)):
        cv_results = cross_val_score(regressors[i], combinedTrain, y_train, cv=kfold, scoring=scoring)
        compareDF.loc[i] = [regressor_names[i], -np.mean(cv_results)]
    if verbose == 1:
        sns.barplot(x="Regressors", y="Validation_Score", data=compareDF)
    elif verbose == 2:
        print compareDF
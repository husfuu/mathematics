import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_missings_persentation(df):
    """print missing values each each variables in the dataframe

    Args:
        df (_dataframe_): dataframe that u want to check missing values

    Returns:
        _dataframe_
    """
    labels,values = list(),list()
    if df.isna().sum().sum()>0:
        for column in df.columns:
            if df[column].isnull().sum():
                labels.append(column)
                values.append((df[column].isnull().sum() / len(df[column]))*100)
        missings=pd.DataFrame({'Feature':labels,'MissingPercent':values }).sort_values(by='MissingPercent',ascending=False)
        return missings
    else:
        return False


def cleanData(result_df): 
    """fix the history data (model results)

    Args:
        result_df (_dataframe_): history data model
    """
    for i in result_df.columns:
        column_names = i.split("_")
        if (len(column_names) > 2) and (column_names[-1] == '1'):
            result_df.rename(columns = {i: i.split("_")[0] + '_' + i.split("_")[1]}, inplace = True)
        elif (len(column_names) <= 2) and (column_names[-1] == '1'):  
            result_df.rename(columns = {i: i.split("_")[0]}, inplace = True)


def addEpochs(history_model):
    """_summary_

    Args:
        history_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    epochs = np.arange(0, 200, 1, dtype=int)
    history_model['epochs'] = epochs

    return history_model


def finalResultsMetrics(result_df):
    """

    Args:
        result_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = dict()
    col_names = list(result_df.columns)
    for col_name in col_names:
        if col_name == 'Unnamed: 0':
            continue
        res[col_name] = result_df[col_name].iloc[-1]
    
    return res

def plotResults(result_df):
  sns.set(rc = {'figure.figsize':(15, 8)})

  figADAM, axesADAM = plt.subplots(2, 2)

  # Accuracy 
  resAdamAccuracy = pd.melt(result_df[['val_accuracy', 'accuracy', 'epochs']], ['epochs'])
  sns.lineplot(x='epochs', y='value', hue='variable', data=resAdamAccuracy, ax=axesADAM[0, 0])

  # Precision
  resAdamPrecision = pd.melt(result_df[['val_precision', 'precision', 'epochs']], ['epochs'])
  sns.lineplot(x='epochs', y='value', hue='variable', data=resAdamPrecision, ax=axesADAM[0, 1])

  # Recall
  resAdamRecall = pd.melt(result_df[['val_recall', 'recall', 'epochs']], ['epochs'])
  sns.lineplot(x='epochs', y='value', hue='variable', data=resAdamRecall, ax=axesADAM[1, 0])

  # AUC
  resAdam_AUC = pd.melt(result_df[['val_auc', 'auc', 'epochs']], ['epochs'])
  sns.lineplot(x='epochs', y='value', hue='variable', data=resAdam_AUC, ax=axesADAM[1, 1])

  plt.show()
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\Dyche Data Science class\carnava.csv')
df.dtypes
drop_cols = list(df.select_dtypes(['object']).columns)
df = df.drop(drop_cols,axis=1)
df = df.astype(float)
df.dtypes
#how to find ouliers in numeric columns in data set with a simple function
def numberOfOutliers(mySeries, upperOutlier, lowerOutlier):
    return sum((mySeries > upperOutlier.loc[mySeries.name,]) | \
               (mySeries < lowerOutlier.loc[mySeries.name,]))


def outlier_stats(df):
    numericDescribe = (df.describe(include='all').T).round(decimals=3)

    # Calculate outliers using this formula: first quartile – 1.5·IQR > outlier > third quartile + 1.5·IQR
    numericDescribe['IQR'] = numericDescribe['75%'] - numericDescribe['25%']
    numericDescribe['outliers'] = (numericDescribe['max'] > (numericDescribe['75%'] + (1.5 * numericDescribe['IQR']))) \
                                  | (numericDescribe['min'] < (numericDescribe['25%'] - (1.5 * numericDescribe['IQR'])))

    # Calculate IQR for each column of the dataframe.
    IQR = df.quantile(.75) - df.quantile(.25)

    # Calculate the upper and lower outlier values
    upperOutlier = df.quantile(.75) + (1.5 * (IQR))
    lowerOutlier = df.quantile(.25) - (1.5 * (IQR))

    # Store the result in a new column
    numericDescribe['num_outliers'] = df.apply(numberOfOutliers, args=(upperOutlier, lowerOutlier))
    numericDescribe.sort_values('num_outliers', ascending=False, inplace=True)
    newColOrder = ['count', 'outliers', 'num_outliers', 'IQR', 'mean', 'std', \
                   'min', '25%', '50%', '75%', 'max']
    numericDescribe = numericDescribe.reindex(columns=newColOrder)

    return numericDescribe

df_outliers = outlier_stats(df)
print(df_outliers)
#balancing my target data with using SMOTE technic
from imblearn.over_sampling import SMOTE
sm = SMOTE()
seed = 100
k = 1
X = Final_df.loc[:, Final_df.columns != 'Target']
Y = Final_df.Target
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_res, Y_res = sm.fit_resample(X, Y)
df_smote = pd.concat([pd.DataFrame(Y_res), pd.DataFrame(X_res)], axis=1)
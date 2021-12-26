import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn import metrics

data = pd.read_csv("method.csv")

def mapNumberOfBugs(row):
    return 1 if row['Number of Bugs'] > 0 else 0

data['Number of Bugs'] = data.apply(mapNumberOfBugs, axis=1)

data = data.drop(columns=['Project','Hash', 'LongName', 'Vulnerability Rules', 'Finalizer Rules', 'Migration15 Rules', 'Migration14 Rules', 'Migration13 Rules', 'MigratingToJUnit4 Rules', 'JavaBean Rules', 'Coupling Rules', 'WarningBlocker', 'Code Size Rules', 'WarningInfo', 'Android Rules', 'Clone Implementation Rules', 'Comment Rules'])
X=data.iloc[:,0:58].values
Y=data.iloc[:,58:59].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

catBoostModel = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass')

catBoostModel.fit(X_train,Y_train)
Y_pred=catBoostModel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
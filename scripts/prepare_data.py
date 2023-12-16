import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, "datasets/churn.csv"))

numbCol = ['EstimatedSalary', 'Balance', 'CreditScore','Age']
nomCol = ['HasCrCard', 'IsActiveMember', 'Geography','Gender', 'NumOfProducts', 'Tenure']

df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)
df['Balance'] = df['Balance'].astype(int)

df.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], inplace=True)

#Label Encoding
le = LabelEncoder()
lstforle = ['Geography', 'Gender']
for i in lstforle :
    df[i] = le.fit_transform(df[i])
    print(i,' : ',df[i].unique(),' = ',le.inverse_transform(df[i].unique()))

sc = StandardScaler()
for col in numbCol:
    df[col] = sc.fit_transform(df[[col]])
df.head()
df.drop(columns = ['HasCrCard'], inplace=True)

df.to_csv(os.path.join(BASE_DIR, "datasets/churn_prepared.csv"))
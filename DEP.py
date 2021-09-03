import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_excel('car data.xlsx')
for i in range(len(df1.Year)):
    df1.loc[i, 'Car_Age'] = 2021-df1.loc[i, 'Year']

colnames = list(df1.columns.values)
df = df1[colnames[0:1] + colnames[5:9] + colnames[1:2] + colnames[4:5] + colnames[9:10] + colnames[3:4] + colnames[2:3]]

df[['Car_Name']] = df[['Car_Name']].astype(str)
df[['Fuel_Type']] = df[['Fuel_Type']].astype(str)
df[['Seller_Type']] = df[['Seller_Type']].astype(str)
df[['Transmission']] = df[['Transmission']].astype(str)
df[['Owner']] = df[['Owner']].astype(str)

df.drop(columns=['Year'], inplace=True)
df_col = df.columns

from sklearn import preprocessing
en = preprocessing.LabelEncoder()
for i in range(len(df_col)):
    if df[df_col[i]].dtype=='object':
        df[df_col[i]] = en.fit_transform(df[df_col[i]])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
stdscale = StandardScaler()    #Gave better results
minmax = MinMaxScaler()
for i in range(len(df_col)):
    df[[df_col[i]]] = stdscale.fit_transform(df[[df_col[i]]])

X = df.iloc[:, 0:8].values
Y = df.iloc[:, 8].values.reshape(-1, 1)

X = np.delete(X, 0, 1) #DROPPING COLUMN 0 (1st) DUE TO HIGH CORRELATION

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

pickle.dump(linreg, open('model.pkl', 'wb'))    #Pickle file of the model
model = pickle.load(open('model.pkl', 'rb'))    #Object of that pickle file of the model
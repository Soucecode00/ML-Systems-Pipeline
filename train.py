import pandas as pd 
df = pd.read_csv('data.csv')
# print(df.head(10))
x = df[['age', 'income', 'savings']]
y = df['approved']

# print('Features')
# print(x.head())

# print('\nTarget column')
# print(y.head())

# print(x.shape)
# print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)
# print('X_train shape:', X_train.shape)
# print('\n X_test shape:', X_test.shape)
# print('\n y_train shape:', y_train.shape)
# print('\n y_test shape:', y_test.shape)##

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(X_train, y_train)
print('model trained successfully')
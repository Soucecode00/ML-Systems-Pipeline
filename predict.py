import joblib 
import numpy as np 
import pandas as pd 
model = joblib.load('model.pkl')
print('model loaded successfully')

new_customer = pd.DataFrame(
    [[30, 40000, 10000]],
    columns=['age', 'income', 'savings']
)
prediction = model.predict(new_customer)
prob = model.predict_proba(new_customer)
print(prediction)
print(prob)
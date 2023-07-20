import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv('database.csv')


data['Test Type'] = data['Test Type'].astype(float)
print(data.dtypes)
features=['Test Type','Technology','Signal Strength']
target=['Data Speed(Kbps)']



x=data[features]
y=data[target]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.fit(x_test)

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

r_squared=model.score(x_test,y_test)

print("mean Sraured error",mse)
print("root mean square error",rmse)
print("R-squraed score",r_squared)
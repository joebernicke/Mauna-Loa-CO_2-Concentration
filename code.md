```python
import pandas as pd
import numpy as np
import os
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
```


### Here I am setting up the data for our regression.
```python
colnames=["Yr","Month","Date1","Date2","CO2","x","y","e","f","g"]
keepcols=["Yr","Month","CO2"]
df=(pd.read_csv('CO2.csv',skiprows=57,header=None,names=colnames).drop(columns=[col for col in colnames if col not in keepcols]).assign(CO2=lambda df: df.CO2.replace({-99.99: np.nan})).assign(time=lambda df: (df.index.values + 0.5) / 12))
df=df.dropna()

x=np.array(df.time.values)
y=np.array(df.CO2.values)
```

### Now, in order to test to see how well we did we need to split the data into a
### training set and a test set.

```python
train_size = int(len(x)*0.8)
gap = 0 # int(len(x)*0.1)
test_size = len(x) - train_size - gap
x_train = x[0:train_size]
y_train = y[0:train_size]
x_test = x[train_size + gap: None]
y_test = y[train_size + gap: None]
```
### Here I'm fitting the Fitting with both old fashioned linear regression and scikits linear regression
#### I'll use scipy LinearRegression first. 
```python
model = LinearRegression().fit(np.array(x_train).reshape(-1, 1),y_train)
coefficients = [model.coef_[0],model.intercept_]
print("The linear model is F(t) = " +str(coefficients[0])+"*t + (" +str(coefficients[1])+")")

#### Here ill do it the old fashion way by finding the Least square error estimator
meanx=np.mean(x_train)
meany=np.mean(y_train)
covar_xy=np.cov(x_train,y_train)[0][1]
varx_train=np.var(x_train,ddof=1)
beta=covar_xy/varx_train
beta0=meany-beta*np.mean(x_train)
```

### Now I'm plotting the predicted values of our regression coefficients to see how  well our line matches the data.  We clearly see our data is not linear.
```python
plt.scatter(x_train,y_train,s=1)
plt.plot(x_train,beta*x_train+beta0)
plt.show()
```


### Now we assess how we did using the Residuals, RMSE, and MAPE on the test set. Again I'll do it using a sclearn package and the old fashion way

```python
residuals=y_train-(beta*x_train+beta0)

y_pred_train=beta*x_train+beta0

rmse=(np.sum([i**2/len(y_test) for i in y_test-(beta*x_test+beta0)]))**0.5
rmse1=mean_squared_error(y_test,beta*x_test+beta0)**0.5
print(rmse,rmse1)

y_pred_test=beta*x_test+beta0
mape=np.sum([np.abs(i) for i in (y_test-y_pred_test)/y_test])/len(y_test)
mape1=mean_absolute_percentage_error(y_test,y_pred_test)
print(mape,mape1)

### We see just as expected, the errors are clearly not random.
plt.scatter(x_train,residuals)
plt.show()
```


### Since our data isn't linear, lets try fitting to a quadratic. Same as before, I'll do it using a math package and I'll do it manually to verify
```python

x_train_squared=x_train**2
X=np.c_[x_train_squared,x_train,np.ones(len(x_train))]

beta=np.linalg.inv(X.T @ X) @ X.T @ y_train

degree=2
model_2=make_pipeline(PolynomialFeatures(degree),LinearRegression())
model_2.fit(np.array(x_train).reshape(-1, 1),y_train)
quadratic = model_2.predict(np.array(x_train).reshape(-1, 1))

coefficients_2 = [model_2.named_steps.linearregression.coef_[-1],model_2.named_steps.linearregression.coef_[-2],model_2.named_steps.linearregression.intercept_]
print("The quadratic model is F(t) = " +str(coefficients_2[0])+"*t^2 + (" +str(coefficients_2[1])+")*t + (" +str(coefficients_2[2])+")")
print(str(beta)+' We get the same answer as above')
```

### We now do the same as before. Assess our new quadratic regression estimates with Residuals, RMSE, and MAPE on the test dataset
```python
#### We first plot residuals to visually check if there is any trend in the errors There aren't so we are good.
residuals=y_train-beta @ X.T 
plt.scatter(X[:,0],residuals)

x_test_squared=x_test**2
X_test=np.c_[x_test_squared,x_test,np.ones(len(x_test))]
rmse=(np.sum([i**2/len(y_test) for i in y_test-beta @ X_test.T]))**0.5
rmse1=mean_squared_error(y_test,beta @ X_test.T)**0.5

mape=np.sum([np.abs(i) for i in (y_test-beta @ X_test.T)/y_test])/len(y_test)
mape1=mean_absolute_percentage_error(y_test,beta @ X_test.T)
```

### Now, we noticed in the original data that incorporated within the Co2 trend there seems to be a periodic element in the intra-year timeframe.  We're going to pull that out.
```python
df_train=df[0:587]
#### Directly below here we give our data an extra column equal to the residuals. This allows us to group-by the month variable, and find the average co2 level for each month across all years without our quadratic trend influencing the averages. 
df_train = df_train.assign(R_train=lambda df_train: (y_train -beta @ X.T))
monthavg=df_train.groupby('Month')['R_train'].mean()

#### Below is a function to create pretty graphs of the average monthly variation. the interp1d function gives
month_continuous = np.linspace(1,12,num = 100, endpoint =True)
periodic= interp1d(monthavg.index,monthavg.values,kind = 'cubic')
plt.figure()
plt.scatter(monthavg.index,monthavg.values)
plt.plot(month_continuous,periodic(month_continuous))
plt.show()
```


### Now we put together the final model. The final model is our Periodic trend (P)+ our quadratic trend (q)
```python
#### We first make a feature matrix with all data, not just the training set
df=df.assign(timesquared=lambda df: df.time**2)
plt.scatter(df['time'],df['CO2'],s=0.24)
X_tot=np.array([df.timesquared,df.time,np.ones(len(df.time))]).T

#### Now we resolve our quadratic predicted values, and then add it to the monthly periodic trend we found
quadratic=beta @ X_tot.T
periodic=monthavg[df.Month]
Total_trend=quadratic+periodic


plt.plot(df.time,Total_trend)
plt.show()
```
### In the end, we have a model that fits.  This means that our regression coefficients are accurate, and if we have another unknown time in the future we can accurately predict what the co2 value will be. The form of the equation is co2 level= Periodic trend(time) + quadratic trend(time+time^2)







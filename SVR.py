import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

path = r'C:\Users\santo\Desktop\ML Course\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv'
df = pd.read_csv(path)

x= df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
y = y.reshape(-1,1)

#feature scaling for SVR
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
a =np.array([[6.5]])
model = SVR(kernel='rbf')
model = model.fit(x,y)
a=sc_x.transform(a)
y_pred = np.array(model.predict(a))
y_pred = y_pred.reshape(-1,1)
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)
b= model.predict(x)
b = b.reshape(-1,1)

x_grid = np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
c= sc_x.transform(x_grid)
c = model.predict(c)
c = c.reshape(-1,1)

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color= 'orange')
plt.plot(x_grid,sc_y.inverse_transform(c), color= 'green')
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color= 'red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(b),color= 'blue')
plt.show()
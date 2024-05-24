# Update1.1: on python code file
# Update2: New line added to both files
# Update3: Last and final update 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


df = pd.read_excel("C:\Andrew\Python\M.L_Dataset_assignment\Concrete_Data.xls", sheet_name='Sheet1')

new_names = ['Cement', 'Slag', 'Ash', 'Water', 'Superplasticizer', 'Coarse', 'Fine', 'Age','Strength']
df.columns = new_names #preprocessing1
#print(df)
#print(df.isna().sum()) #preprocessing2
#print(df.describe()) #preprocessing3

X = df.drop(columns=['Strength'])
Y = df['Strength']
print(X)
'''
plt.hist(Y)
plt.xlabel('Strength')
plt.ylabel('Frequency')
plt.title('Distribution of Strength vs Frequency')
plt.show()
'''

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# finding Errors before scaling

svr = SVR().fit(x_train, y_train)    
y_pred = svr.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Alternatively, use np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)


# using random forest to find feature importance
# from Random forest we find that coulumns (Ash,Coarse, Fine) affect < 4%
# we remove them and find better rmse,mse,mae,r2 scores

model = RandomForestRegressor(n_estimators=100)

rf = model.fit(x_train, y_train)
perm_importance = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

f_importance = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending=False)
print(f_importance)

plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importances")
plt.show()


# finding Errors after applying StandardScaler
'''
scaler = StandardScaler() #preprocessing5.1
x_train_SSscaler = scaler.fit_transform(x_train)
x_test_SSscaler = scaler.transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#preprocessing6

svr = SVR().fit(x_train_SSscaler, y_train)    
y_pred = svr.predict(x_test_SSscaler)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Alternatively, use np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
'''


# finding Errors after applying MinMaxScaler
'''
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

scaler1 = MinMaxScaler() 
x_train_MMscaler = scaler1.fit_transform(x_train)
x_test_MMscaler = scaler1.transform(x_test)

svr = SVR().fit(x_train_MMscaler, y_train)    
y_pred = svr.predict(x_test_MMscaler)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Alternatively, use np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
'''

# finding Errors after dropping low feature importance columns
X_mod = X.drop(columns=['Ash', 'Coarse']) 
x_train, x_test, y_train, y_test = train_test_split(X_mod, Y, test_size=0.15, random_state=42)
scaler = StandardScaler() #preprocessing5.1
x_train_SSscaler = scaler.fit_transform(x_train)
x_test_SSscaler = scaler.transform(x_test)
svr = SVR().fit(x_train_SSscaler, y_train)    
y_pred = svr.predict(x_test_SSscaler)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Alternatively, use np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)


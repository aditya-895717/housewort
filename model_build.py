import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Loading dataset
housing = pd.read_csv("Dataset/Housing.csv")
housing['price'] = housing['price'] / 100000
housing['price'] = housing['price'].astype(float)

# Outlier treatment for 'price' and 'area'
Q1_price = housing['price'].quantile(0.25)
Q3_price = housing['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
housing = housing[(housing['price'] >= Q1_price - 1.5*IQR_price) & (housing['price'] <= Q3_price + 1.5*IQR_price)]

Q1_area = housing['area'].quantile(0.25)
Q3_area = housing['area'].quantile(0.75)
IQR_area = Q3_area - Q1_area
housing = housing[(housing['area'] >= Q1_area - 1.5*IQR_area) & (housing['area'] <= Q3_area + 1.5*IQR_area)]

# Mapping 'yes' and 'no' attributes to 1 and 0
yes_no_attributes = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
mapping = {'yes': 1, 'no': 0}
for x in yes_no_attributes:
    housing[x] = housing[x].map(mapping)

# Converting 'furnishingstatus' to string and perform one-hot encoding
housing['furnishingstatus'] = housing['furnishingstatus'].astype(str)
housing = pd.get_dummies(housing, columns=['furnishingstatus'], drop_first=True, dtype=int)

# Scaling numerical variables
scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing[num_vars] = scaler.fit_transform(housing[num_vars])

# Spliting data into train and test sets
X = housing.drop('price', axis='columns')
y = housing['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Regression
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Saving model and scaler

# Normalize MSE (lower is better, so we invert it)
mse_lr_normalized = 1 / (1 + mse_lr)  # Inverting MSE for normalization
mse_rf_normalized = 1 / (1 + mse_rf)

# Normalize R² (higher is better)
r2_lr_normalized = r2_lr
r2_rf_normalized = r2_rf

# Combining scores
score_lr = 0.5 * mse_lr_normalized + 0.5 * r2_lr_normalized
score_rf = 0.5 * mse_rf_normalized + 0.5 * r2_rf_normalized

# Model selection based on combined scores
if score_lr > score_rf:
    best_model = "Linear Regression"
    best_mse = mse_lr
    best_r2 = r2_lr
else:
    best_model = "Random Forest Regression"
    best_mse = mse_rf
    best_r2 = r2_rf

print(f"\nBest Model: {best_model}")
print(f"Best Model MSE: {best_mse}")
print(f"Best Model R²: {best_r2}")

# Saving the best model
if best_model == "Linear Regression":
    joblib.dump(model_lr, 'ML_Models/model.joblib')
else:
    joblib.dump(model_rf, 'ML_Models/model.joblib')

# Saving the scaler used for later use
joblib.dump(scaler, 'ML_Models/scaler.joblib')
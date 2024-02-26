import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('real_estate_data.csv')

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.describe())

# Handling missing data
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Visualize the distribution of features
plt.figure(figsize=(10, 6))
for i, col in enumerate(data.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data_imputed[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Outlier detection and handling
plt.figure(figsize=(10, 6))
for i, col in enumerate(data_imputed.columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data_imputed[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Based on the boxplots, let's clip outliers in the 'price' column
data_imputed['price'] = data_imputed['price'].clip(upper=data_imputed['price'].quantile(0.95))

# Visualize the relationship between features and target variable
sns.pairplot(data_imputed)
plt.show()

# Preprocessing: Assuming 'size' and 'number of bedrooms' as features
X = data_imputed[['size', 'num_bedrooms']]
y = data_imputed['price']

# Feature engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Feature selection using RandomForestRegressor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
feature_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
X_selected = feature_selector.fit_transform(X_scaled, y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model ensemble: Ridge Regression and Random Forest
ridge_params = {'alpha': [0.1, 1, 10, 100]}
ridge = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluating the models
models = {'Ridge Regression': ridge, 'Random Forest': rf}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Test Set Mean Squared Error: {mse}")
    print(f"{name} - R-squared Score: {r2}")

# Example usage: Predicting the price for a new real estate property
new_property = [[2000, 3]]  # size: 2000 sqft, 3 bedrooms
new_property_poly = poly.transform(new_property)
new_property_scaled = scaler.transform(new_property_poly)
new_property_selected = feature_selector.transform(new_property_scaled)

ridge_predicted_price = ridge.predict(new_property_selected)
rf_predicted_price = rf.predict(new_property_selected)

print("Ridge Regression - Predicted Price for the New Property:", ridge_predicted_price)
print("Random Forest - Predicted Price for the New Property:", rf_predicted_price)

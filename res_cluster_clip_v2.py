import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load cleaned data
test_data = pd.read_pickle('cleaned_test_data_v2.pkl')
train_data = pd.read_pickle('cleaned_train_data_v2.pkl')

# Load the saved model
model = tf.keras.models.load_model('optimized_final_resnet_model.h5')
print('Model loaded: optimized_final_resnet_model.h5')

# Selected features based on SHAP analysis
selected_features = [
    'arf', 'depreciation', 'car_age', 'power', 'coe', 'road_tax', 'mileage', 
    'calculated_price', 'dereg_value_normalized', 'power_to_weight', 'engine_cap'
]

# Retain useful interaction features based on related features
interaction_features = [col for col in train_data.columns if 'interaction' in col]
selected_features += interaction_features

# Preprocessing function for data
def preprocess_data(data, imputer, scaler):
    # Apply the imputer and scaler to the data
    X = imputer.transform(data[selected_features])
    X_normalized = scaler.transform(X)
    return X_normalized

# Handle missing values by imputing with mode (most frequent)
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(train_data[selected_features])

# Standardize the features
scaler = StandardScaler()
scaler.fit(train_data[selected_features])

# Preprocess training data for regression model
X_train_limits = preprocess_data(train_data, imputer, scaler)
y_train_min = train_data['price'].groupby([train_data['make_encoded'], train_data['model_encoded']]).transform('min')
y_train_max = train_data['price'].groupby([train_data['make_encoded'], train_data['model_encoded']]).transform('max')

# Step 1: Train regression models to predict price limits
min_price_model = LinearRegression()
max_price_model = LinearRegression()
min_price_model.fit(X_train_limits, y_train_min)
max_price_model.fit(X_train_limits, y_train_max)

# Preprocess the test data
X_test = preprocess_data(test_data, imputer, scaler)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 2: Predict price limits for each row in the test set
predicted_price_min = min_price_model.predict(X_test.reshape(X_test.shape[0], -1))
predicted_price_max = max_price_model.predict(X_test.reshape(X_test.shape[0], -1))

test_data['price_min'] = predicted_price_min
test_data['price_max'] = predicted_price_max

# Step 3: Predict on test data and clip predictions based on regression model predicted limits
y_test_pred = model.predict(X_test).flatten()
y_test_pred_clipped = np.clip(y_test_pred, a_min=test_data['price_min'], a_max=test_data['price_max'])

# Prepare the submission file
submission = pd.DataFrame({
    'Id': test_data.index,
    'Predicted': y_test_pred_clipped
})
submission = submission.sort_values(by='Id')
submission.to_csv('submission_with_regression_limits.csv', index=False)
print('Submission file generated: submission_with_regression_limits.csv')

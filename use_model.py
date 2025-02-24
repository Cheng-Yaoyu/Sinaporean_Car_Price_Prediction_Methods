import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load cleaned training and test data
train_data = pd.read_pickle('cleaned_train_data_v2.pkl')
test_data = pd.read_pickle('cleaned_test_data_v2.pkl')

# Preprocessing function for data (re-fitting imputer and scaler)
def preprocess_data_for_prediction(train_data, test_data):
    selected_features = [
        'arf', 'depreciation', 'car_age', 'power', 'coe', 'road_tax', 'mileage', 
        'calculated_price', 'dereg_value_normalized', 'power_to_weight', 'engine_cap'
    ]

    # Retain useful interaction features based on related features
    interaction_features = [col for col in train_data.columns if 'interaction' in col]
    selected_features += interaction_features

    # Handle missing values by imputing with mode (most frequent)
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(train_data[selected_features])  # Fit imputer using train_data

    # Apply the imputer to both training and test data
    X_train = imputer.transform(train_data[selected_features])
    X_test = imputer.transform(test_data[selected_features])

    # Standardize the features
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler using train data

    # Apply the scaler to both training and test data
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized, imputer, scaler

# Apply preprocessing to train and test data
X_train_normalized, X_test_normalized, imputer, scaler = preprocess_data_for_prediction(train_data, test_data)

# Reshape input data for Conv1D model
X_test_normalized = X_test_normalized.reshape((X_test_normalized.shape[0], X_test_normalized.shape[1], 1))

# Load the tuned ResNet model
final_model = tf.keras.models.load_model('optimized_final_resnet_model.h5')

# Predict on test data
y_test_pred = final_model.predict(X_test_normalized).flatten()

# Clip predictions to prevent extreme values (based on training target range)
y_test_pred = np.clip(y_test_pred, a_min=train_data['price'].min(), a_max=train_data['price'].max())

# Prepare the submission file
submission = pd.DataFrame({
    'Id': test_data.index,
    'Predicted': y_test_pred
})
submission = submission.sort_values(by='Id')
submission.to_csv('submission.csv', index=False)

print('Submission file generated: submission.csv')

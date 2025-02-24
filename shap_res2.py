import shap
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load cleaned data
train_data = pd.read_pickle('cleaned_train_data_v2.pkl')
test_data = pd.read_pickle('cleaned_test_data_v2.pkl')

# Step 1: Create additional features
train_data['depreciation_age'] = train_data['depreciation'] * train_data['car_age']
train_data['make_model_depreciation_age'] = train_data['make_encoded'] * train_data['model_encoded'] * train_data['depreciation'] * train_data['car_age']

test_data['depreciation_age'] = test_data['depreciation'] * test_data['car_age']
test_data['make_model_depreciation_age'] = test_data['make_encoded'] * test_data['model_encoded'] * test_data['depreciation'] * test_data['car_age']

def preprocess_data(data):
    selected_features = [
        'arf', 'depreciation', 'car_age', 'power', 'coe', 'road_tax', 'mileage', 'omv',
        'power_to_weight', 'engine_cap','make_encoded', 'model_encoded',
        'calculated_price', 'dereg_value_normalized', 'depreciation_age', 'make_model_depreciation_age'
    ]

    # Include SVD features, embedding features, and interaction features
    svd_features = [col for col in data.columns if '_svd_' in col]
    embed_features = [col for col in data.columns if '_embed_' in col]
    interaction_features = [col for col in data.columns if 'interaction' in col]

    # Adding the SVD, embedding, and interaction features
    selected_features += svd_features + embed_features + interaction_features

    # Handle missing values by imputing with mode (most frequent)
    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(data[selected_features])
    y = data['price']  # Use the original target variable

    # Standardize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y, imputer, scaler, selected_features

# Preprocess the data
X, y, imputer, scaler, selected_features = preprocess_data(train_data)

# Load the trained ResNet model
final_model = tf.keras.models.load_model('optimized_final_resnet_model.h5')

# SHAP analysis
explainer = shap.Explainer(final_model, X)
shap_values = explainer(X[:3000])  # Calculate SHAP values for the first 3000 samples to reduce computation time

# Map feature indices to feature names
shap_values.feature_names = selected_features

# Plot summary of SHAP values and save it
plt.figure()
shap.summary_plot(shap_values, X[:3000], show=False)
plt.gca().tick_params(labelsize=10)  # Adjust tick label size for better visibility
plt.savefig('shap_summary_plot_named_adjusted.png', dpi=300)

# Plot bar chart of SHAP values and save it
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.gca().tick_params(labelsize=10)  # Adjust tick label size for better visibility
plt.savefig('shap_bar_plot_named_adjusted.png', dpi=300)

print("SHAP plots saved successfully with adjusted font size.")

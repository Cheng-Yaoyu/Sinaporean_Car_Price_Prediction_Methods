import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load cleaned data
train_data = pd.read_pickle('cleaned_train_data_v2.pkl')
test_data = pd.read_pickle('cleaned_test_data_v2.pkl')
'''
# Step 1: Create additional features
train_data['depreciation_age'] = train_data['depreciation'] * train_data['car_age']
train_data['make_model_depreciation_age'] = train_data['make_encoded'] * train_data['model_encoded'] * train_data['depreciation'] * train_data['car_age']

test_data['depreciation_age'] = test_data['depreciation'] * test_data['car_age']
test_data['make_model_depreciation_age'] = test_data['make_encoded'] * test_data['model_encoded'] * test_data['depreciation'] * test_data['car_age']
'''
# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Preprocessing function for training data
def preprocess_data(data):
    # Selected features based on SHAP analysis
    selected_features = [
        'arf', 'depreciation', 'car_age', 'power', 'coe', 'road_tax', 'mileage', 
        'calculated_price', 'dereg_value_normalized', 'power_to_weight', 'engine_cap'
    ]

    # Retain useful interaction features based on related features
    interaction_features = [col for col in data.columns if 'interaction' in col]
    selected_features += interaction_features

    # Handle missing values by imputing with mode (most frequent)
    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(data[selected_features])
    y = data['price']  # Use the original target variable

    # Standardize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y, imputer, scaler

# Preprocess the data
X, y, imputer, scaler = preprocess_data(train_data)

# Define ResNet block
def resnet_block(input_layer, filters, kernel_size=3):
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, input_layer])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Define ResNet model
def build_resnet_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim, 1))
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add multiple ResNet blocks
    for _ in range(2):  # 可以减少或增加 block 数量
        x = resnet_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Reshape input data for Conv1D
X = X.reshape((X.shape[0], X.shape[1], 1))

# Set up cross-validation using KFold for regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 512
epochs = 1000  # 减少 epoch，避免过拟合
rmse_list = []

# Define ReduceLROnPlateau for fold training
reduce_lr_fold = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_root_mean_squared_error',
    factor=0.5,
    patience=10,  # 减少 patience，避免浪费过多 epoch
    min_lr=1e-6,
    verbose=1
)

# Define EarlyStopping for fold training
early_stopping_fold = tf.keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=20,  # 减少 patience，适时停止
    min_delta=0.01,
    verbose=1
)

# Define ReduceLROnPlateau for final training
reduce_lr_final = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='root_mean_squared_error',  # Monitor training loss during final training
    factor=0.5,         # Reduce learning rate by 50% when training loss does not improve
    patience=10,        # Number of epochs to wait before reducing learning rate
    min_lr=1e-6,        # Minimum learning rate
    verbose=1           # Output learning rate reduction information
)

# Define EarlyStopping for final training
early_stopping_final = tf.keras.callbacks.EarlyStopping(
    monitor='root_mean_squared_error',  # Monitor training loss during final training
    patience=20,         # Number of epochs to wait before stopping
    min_delta=0.01,      # Minimum change to qualify as an improvement
    verbose=1            # Output stopping information
)

# Cross-validation training
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Build and train the ResNet model
    model = build_resnet_model(X_train.shape[1])
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[reduce_lr_fold, early_stopping_fold],
                        verbose=1)

    # Predict on validation set and calculate RMSE
    y_val_pred = model.predict(X_val).flatten()
    y_val_pred = np.clip(y_val_pred, a_min=y_val.min(), a_max=y_val.max())
    rmse = calculate_rmse(y_val, y_val_pred)
    rmse_list.append(rmse)
    print(f'Fold RMSE: {rmse}')

# Calculate average RMSE across folds
average_rmse = np.mean(rmse_list)
print(f'Average Validation RMSE: {average_rmse}')

def preprocess_test_data(data, imputer, scaler):
    selected_features = [
        'arf', 'depreciation', 'car_age', 'power', 'coe', 'road_tax', 'mileage', 
        'calculated_price', 'dereg_value_normalized', 'power_to_weight', 'engine_cap'
    ]

     # Retain useful interaction features based on related features
    interaction_features = [col for col in data.columns if 'interaction' in col]
    selected_features += interaction_features

    # Apply the imputer and scaler to the test data
    X = imputer.transform(data[selected_features])
    X_normalized = scaler.transform(X)
    
    return X_normalized

X_test = preprocess_test_data(test_data, imputer, scaler)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train final model on the full dataset
final_model = build_resnet_model(X.shape[1])
history = final_model.fit(X, y, 
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=[reduce_lr_final, early_stopping_final],
                          verbose=1)

# Save the final trained model
final_model.save('optimized_final_resnet_model.h5')
print('Model saved as optimized_final_resnet_model.h5')

# Predict on test data
y_test_pred = final_model.predict(X_test).flatten()
y_test_pred = np.clip(y_test_pred, a_min=train_data['price'].min(), a_max=train_data['price'].max())  # Clip predictions to prevent extreme values

# Prepare the submission file
submission = pd.DataFrame({
    'Id': test_data.index,
    'Predicted': y_test_pred
})
submission = submission.sort_values(by='Id')
submission.to_csv('submission.csv', index=False)
print('Submission file generated: submission.csv')

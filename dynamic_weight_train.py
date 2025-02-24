import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the datasets
file1 = 'train.csv'
file2 = 'predict_train_1.csv'
file3 = 'predict_train_2.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Merge the DataFrames on 'Id'
merged_df = pd.merge(df1, df2, on='Id', suffixes=('', '_1'))
merged_df = pd.merge(merged_df, df3, on='Id', suffixes=('', '_2'))

# Ensure predictions are numeric
merged_df['Predicted_1'] = pd.to_numeric(merged_df['Predicted_1'], errors='coerce')
merged_df['Predicted_2'] = pd.to_numeric(merged_df['Predicted_2'], errors='coerce')
merged_df['Predicted'] = pd.to_numeric(merged_df['Predicted'], errors='coerce')

# Drop rows with NaN values in predictions
filtered_df = merged_df.dropna(subset=['Predicted_1', 'Predicted_2', 'Predicted']).copy()

# Function to calculate RMSE given a weight
def calculate_weighted_rmse(weight, true_values, pred1, pred2):
    combined_pred = weight * pred1 + (1 - weight) * pred2
    return np.sqrt(mean_squared_error(true_values, combined_pred))

# Function to find the best weight for each price range
def find_best_weight(data, pred1_col, pred2_col, true_col):
    weights = np.linspace(0, 1, 201)  # Using finer weight steps
    best_rmse = float('inf')
    best_weight = None

    for weight in weights:
        rmse = calculate_weighted_rmse(weight, data[true_col], data[pred1_col], data[pred2_col])
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = weight

    return best_weight, best_rmse

# Grid search with finer quantile steps and narrower range around initial boundaries
best_config = None
lowest_rmse = float('inf')

for low_q in np.arange(0.35, 0.41, 0.005):  # Narrowed around initial boundary with finer steps
    for high_q in np.arange(0.55, 0.61, 0.005):  # Narrowed around initial boundary with finer steps
        if low_q >= high_q:
            continue  # Ensure low_q < high_q

        # Calculate dynamic price thresholds for this configuration
        low_price_threshold = filtered_df['Predicted'].quantile(low_q)
        high_price_threshold = filtered_df['Predicted'].quantile(high_q)

        # Split the data into three price ranges
        low_price_data = filtered_df[filtered_df['Predicted'] < low_price_threshold]
        mid_price_data = filtered_df[(filtered_df['Predicted'] >= low_price_threshold) & (filtered_df['Predicted'] <= high_price_threshold)]
        high_price_data = filtered_df[filtered_df['Predicted'] > high_price_threshold]

        # Find best weights for each price range
        low_best_weight, _ = find_best_weight(low_price_data, 'Predicted_1', 'Predicted_2', 'Predicted')
        mid_best_weight, _ = find_best_weight(mid_price_data, 'Predicted_1', 'Predicted_2', 'Predicted')
        high_best_weight, _ = find_best_weight(high_price_data, 'Predicted_1', 'Predicted_2', 'Predicted')

        # Apply the best weights dynamically based on this configuration
        def apply_dynamic_weight(row):
            if row['Predicted'] < low_price_threshold:
                return low_best_weight
            elif row['Predicted'] > high_price_threshold:
                return high_best_weight
            else:
                return mid_best_weight

        filtered_df.loc[:, 'Weight'] = filtered_df.apply(apply_dynamic_weight, axis=1)
        filtered_df.loc[:, 'Final_Predicted'] = (
            filtered_df['Weight'] * filtered_df['Predicted_1'] +
            (1 - filtered_df['Weight']) * filtered_df['Predicted_2']
        )

        # Calculate the overall RMSE for the final predictions
        overall_rmse = np.sqrt(mean_squared_error(filtered_df['Predicted'], filtered_df['Final_Predicted']))

        # Update the best configuration if the current one is better
        if overall_rmse < lowest_rmse:
            lowest_rmse = overall_rmse
            best_config = {
                'low_q': low_q,
                'high_q': high_q,
                'low_weight': low_best_weight,
                'mid_weight': mid_best_weight,
                'high_weight': high_best_weight,
                'overall_rmse': overall_rmse
            }

# Output the best configuration
print(f"Best Configuration: Low quantile = {best_config['low_q']}, High quantile = {best_config['high_q']}")
print(f"Low price weight = {best_config['low_weight']}, Mid price weight = {best_config['mid_weight']}, High price weight = {best_config['high_weight']}")
print(f"Overall RMSE = {best_config['overall_rmse']}")

# Apply the best weights based on the best configuration
low_price_threshold = filtered_df['Predicted'].quantile(best_config['low_q'])
high_price_threshold = filtered_df['Predicted'].quantile(best_config['high_q'])

def apply_dynamic_weight(row):
    if row['Predicted'] < low_price_threshold:
        return best_config['low_weight']
    elif row['Predicted'] > high_price_threshold:
        return best_config['high_weight']
    else:
        return best_config['mid_weight']

filtered_df.loc[:, 'Weight'] = filtered_df.apply(apply_dynamic_weight, axis=1)
filtered_df.loc[:, 'Final_Predicted'] = (
    filtered_df['Weight'] * filtered_df['Predicted_1'] +
    (1 - filtered_df['Weight']) * filtered_df['Predicted_2']
)

# Create a submission DataFrame
submission = pd.DataFrame({'Id': df3['Id'], 'Predicted': df3['Predicted']})

# Update submission with weighted predictions where available
submission.update(filtered_df[['Id', 'Final_Predicted']].rename(columns={'Final_Predicted': 'Predicted'}))

# Save the submission file
submission['Predicted'] = submission['Predicted'].round(3)
submission.to_csv('train_submission_optimized_weighted.csv', index=False)
print('Submission file generated: train_submission_optimized_weighted.csv')

import pandas as pd
import numpy as np

# Load the prediction datasets
file2 = 'submission67.csv'  
file3 = 'submission68.csv'  
df1 = pd.read_csv(file2)
df2 = pd.read_csv(file3)

# Ensure predictions are numeric
df1['Predicted'] = pd.to_numeric(df1['Predicted'], errors='coerce')
df2['Predicted'] = pd.to_numeric(df2['Predicted'], errors='coerce')

# Drop rows with NaN values in predictions
filtered_df = pd.merge(df1, df2, on='Id', suffixes=('_1', '_2')).dropna(subset=['Predicted_1', 'Predicted_2']).copy()

# Set default weights for low, mid, and high price ranges
low_price_weight = 0.77
mid_price_weight = 0.995
high_price_weight = 0.9

# Calculate dynamic price thresholds based on the quantiles from the previous step
low_price_threshold = filtered_df['Predicted_1'].quantile(0.355)
high_price_threshold = filtered_df['Predicted_1'].quantile(0.57)

# Apply the appropriate weight based on price range
def apply_dynamic_weight(row):
    if row['Predicted_1'] < low_price_threshold:
        return low_price_weight
    elif row['Predicted_1'] > high_price_threshold:
        return high_price_weight
    else:
        return mid_price_weight

# Add a 'Weight' column to the DataFrame
filtered_df['Weight'] = filtered_df.apply(apply_dynamic_weight, axis=1)

# Calculate the final predicted value using the weighted average
filtered_df['Final_Predicted'] = (
    filtered_df['Weight'] * filtered_df['Predicted_1'] +
    (1 - filtered_df['Weight']) * filtered_df['Predicted_2']
)

# Create a submission DataFrame
submission = pd.DataFrame({'Id': df1['Id'], 'Predicted': df1['Predicted']})

# Update submission with weighted predictions where available
submission.update(filtered_df[['Id', 'Final_Predicted']].rename(columns={'Final_Predicted': 'Predicted'}))

# Save the submission file
submission['Predicted'] = submission['Predicted'].round(3)
submission.to_csv('submission_optimized_weighted.csv', index=False)
print('Submission file generated: submission_optimized_weighted.csv')

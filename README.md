# CS5228 Project Guidlines

## Environment Setup Instructions

To set up the environment for this code, please follow these steps:

1. **Create a Virtual Environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # Linux/MacOS
   myenv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**
   Run the following command to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Key Dependencies Overview**

   - **Data Processing**: `pandas`, `numpy`, `scikit-learn`
   - **Visualization**: `matplotlib`, `seaborn`, `sweetviz`
   - **Encoding and NLP**: `category-encoders`, `scipy`, `sentence-transformers`
   - **Machine Learning Models**: `lightgbm`, `xgboost`, `catboost`, `imbalanced-learn`
   - **Deep Learning**: `tensorflow==2.13.0`, `tensorrt==8.6.1.6`
   - **Parallel Processing**: `joblib`
   - **Explainability**: `shap`

**Note**: There may be additional packages required or version compatibility issues. For any issues, please contact yaoyu@u.nus.edu.

## Exploratory Data Analysis (EDA) Overview

This section explains the steps taken to analyze the dataset and uncover initial insights:

1. **Load Libraries and Data**
   - Key libraries for data processing, visualization, and statistical analysis are loaded.
   - The dataset is imported and basic information (shape, first rows) is displayed for an initial overview.

2. **Data Overview**
   - `describe()` and `info()` provide summary statistics and data type information.
   - Checks for missing values and outliers are done using z-scores.

3. **Missing Values and Outlier Analysis**
   - Missing values in critical columns (e.g., `depreciation`, `mileage`) are examined.
   - Missing ratios by brand are calculated and visualized, including count and ratio of missing values.

4. **Target Variable Distribution**
   - The `price` distribution is analyzed with histogram plots and fitted against several statistical distributions, including Normal, Johnson SU, and Log Normal.
   - Skewness and kurtosis are calculated to understand distribution characteristics.

5. **Feature Analysis**
   - **Numerical Features**: Correlation matrix and skewness/kurtosis values are analyzed for all numerical columns. Histograms and pairplots visualize feature distributions and relationships.
   - **Categorical Features**: Key features (`make`, `model`) are filtered based on occurrence thresholds. Boxplots, violin plots, and count plots explore the price distribution across categories.

6. **Data Profiling Report**
   - A Sweetviz report is generated for a comprehensive, interactive summary of the dataset and saved as `train_data_sweetviz_report.html`.



## Data Cleaning and Feature Engineering Overview

This section details the data cleaning and feature engineering steps applied, based on the latest version from `train_resnet_new.ipynb`.

1. **Data Loading**
   - Utilizes parallel processing to load `train.csv` and `test.csv` datasets simultaneously.

2. **Handling Missing Values**
   - **Make and Model**: Missing `make` values are inferred based on similar models.
   - **Manufactured Date**: Filled by grouping by `make` and `model`, then using median values or `KNNImputer` if gaps remain.
   - **Other Key Features**: Uses group median values and `KNNImputer` to fill `engine_cap`, `curb_weight`, and `no_of_owners`.

3. **Calculated Features**
   - **ARF and OMV Calculations**: Computes ARF based on OMV and vice versa, using custom formulas sensitive to price ranges.
   - **COE Calculation**: Determines remaining COE years using registration date and title information.
   - **Depreciation**: Filled using price-based calculations and a linear regression model for residual missing values.
   - **Mileage and Power**: Imputed using linear regression with related features.
   - **Road Tax**: Filled similarly using engine capacity with linear regression.

4. **Text Feature Processing**
   - **TF-IDF and SVD**: Applies TF-IDF vectorization and SVD for dimensionality reduction on text columns.
   - **Sentence Embeddings**: Uses pre-trained embeddings, further reduced with PCA.

5. **Encoding Categorical Features**
   - **Target Encoding**: Applies target encoding on key categorical features like `make` and `model`.
   - **Frequency Encoding**: Adds frequency encoding for selected categorical variables.

6. **Feature Engineering**
   - **Polynomial Features**: Generates interactions between key features using `PolynomialFeatures`.
   - **Data Augmentation**: Augments data for imbalanced classes (e.g., luxury sedans and sports cars).

7. **Target Variable Calculation**
   - **Calculated Price**: A custom function computes an estimated price based on depreciation and COE data.

8. **Normalization**
   - **Standard Scaling**: Normalizes `dereg_value` for consistent scale across models.

9. **Save Cleaned Data**
   - Final cleaned data is saved as `cleaned_train_data.pkl` and `cleaned_test_data.pkl`.
10. **Prepared Cleaned Data**
   - We have already prepared two cleaned data: `cleaned_train_data_v2.pkl` and `cleaned_test_data_v2.pkl`, which could be directly used for training.

## Model Training and Model Selection

### Model Training Script for ResNet

The file `train_resnet.py` contains the code to train a ResNet model for car price prediction. Here’s a brief overview of the process:

1. **Load Cleaned Data**: 
   - Loads pre-processed data from `cleaned_train_data_v2.pkl` and `cleaned_test_data_v2.pkl`.

2. **Feature Engineering**:
   - Additional features (`depreciation_age`, `make_model_depreciation_age`) are created to capture specific interactions that are impactful for price prediction.

3. **Data Preprocessing**:
   - Missing values are handled using a mode-based imputer, and features are standardized.
   - Selected features are based on prior SHAP analysis.

4. **ResNet Model Architecture**:
   - A custom ResNet model with multiple convolutional layers is built to process the structured data.

5. **Cross-Validation with KFold**:
   - The model is trained using 5-fold cross-validation with `ReduceLROnPlateau` and `EarlyStopping` callbacks to improve performance and prevent overfitting.

6. **Training and Saving the Final Model**:
   - After cross-validation, the model is trained on the full dataset and saved as `optimized_final_resnet_model.h5`.

7. **Making Predictions**:
   - Predictions are generated for the test set, and results are saved to `submission.csv`.

### Running the Script
Simply run the script by executing:
```bash
python train_resnet.py
```

### Directly use trained model

The `use_model.py` script uses a pre-trained ResNet model to generate predictions on test data. It loads and preprocesses the data, applies the model, and saves the predictions in `submission.csv`. To run, use:

```bash
python use_model.py
```

### SHAP Analysis for Feature Selection

This script, designed as a supplemental analysis for the ResNet model training, uses SHAP (SHapley Additive exPlanations) to assess feature importance.

1. **Data Preprocessing**:
   - Loads cleaned training data and creates additional features (`depreciation_age` and `make_model_depreciation_age`).
   - Selects relevant features, including SVD, embedding, and interaction features, and standardizes them.

2. **Load Trained ResNet Model**:
   - Loads the pre-trained ResNet model (`optimized_final_resnet_model.h5`) to evaluate feature importance.

3. **SHAP Analysis**:
   - Computes SHAP values for a sample of 3000 data points to quantify each feature's impact on model predictions.
   - Generates a summary plot and a bar plot of SHAP values, adjusting font sizes for better readability.

4. **Save SHAP Plots**:
   - The summary and bar plots are saved as `shap_summary_plot_named_adjusted.png` and `shap_bar_plot_named_adjusted.png`, respectively.

To run this script, execute:
```bash
python shap_analysis.py
```

### Test Model Script (test_model.ipynb)

The `test_model.ipynb` notebook is a streamlined approach to quickly evaluate models for car price prediction. It involves the following steps:

1. **Data Preprocessing**:
   - Specifies categorical and numerical features, applying median imputation for numerical values and one-hot encoding for categorical values.

2. **Pipeline Setup**:
   - Combines preprocessing steps into a single pipeline for efficient data transformation.

3. **Train-Test Split**:
   - Splits the processed data into training and test sets to validate model performance.

4. **Model Selection and Training**:
   - Uses Ridge regression as a preliminary model for fast experimentation.
   - Trains the model on the processed training data and evaluates it on the test set.

5. **Performance Metrics**:
   - Computes MAE, MSE, and RMSE on the test set to assess model accuracy.

## Dynamic Weight Calculation and Application

The dynamic weighting approach involves two scripts: `dynamic_weight_train.py` to find the optimal weights, and `weighted_model.py` to apply those weights for final predictions.

1. **dynamic_weight_train.py**:
   - This script identifies the best weights for different price ranges by using grid search on quantiles and weights, optimizing for RMSE.
   - Based on quantile thresholds, it divides the data into low, mid, and high price ranges, and finds the best weight for each range.
   - After determining the optimal weights, it generates weighted predictions and saves the results to `train_submission_optimized_weighted.csv`. After that, we could plot the picture to compare the predictions with the train set.

2. **weighted_model.py**:
   - Using the weights from `dynamic_weight_train.py`, this script combines predictions from two models (`submission67.csv` and `submission68.csv`).
   - It applies the pre-set weights for each price range, calculates the weighted predictions, and saves the final results to `submission_optimized_weighted.csv`, which should give the same result as `submission69_Best.csv`

To run each script:
```bash
python dynamic_weight_train.py  # Find optimal weights (Already done)
python weighted_model.py        # Generate predictions with optimal weights
```

## Clipping Finetune

The `res_cluster_clip_v2.py` script applies regression-based price limits to refine predictions. It uses a pre-trained ResNet model for initial price predictions and then clips these predictions within minimum and maximum price limits calculated by regression models based on each car's characteristics.

To run:
```bash
python res_cluster_clip_v2.py
```

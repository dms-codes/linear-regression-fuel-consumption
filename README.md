
# Linear Regression Model for CO2 Emissions Prediction

This project implements a linear regression model to predict CO2 emissions based on fuel consumption data. The dataset used in this project is `FuelConsumption.csv`, which contains various features related to fuel consumption and corresponding CO2 emissions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Metrics](#metrics)
- [Visualization](#visualization)
- [License](#license)

## Overview

The goal of this project is to build a linear regression model that predicts CO2 emissions based on fuel consumption data. The project performs the following steps:

1. Load and prepare the dataset.
2. Train a linear regression model using specified features.
3. Evaluate the model's performance using various metrics.
4. Visualize the regression results and evaluation metrics.

## Installation

To run this project, you'll need Python and some specific libraries. It is recommended to create a virtual environment. Follow these steps:

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - For Windows:

     ```bash
     venv\Scripts\activate
     ```

   - For macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**

   ```bash
   pip install pandas matplotlib scikit-learn
   ```

## Usage

1. Ensure that the `FuelConsumption.csv` file is in the same directory as the script or update the path in the script.
2. Run the script:

   ```bash
   python <script_name>.py
   ```

The script will load the dataset, train the linear regression model, evaluate its performance, and visualize the regression results.

## Functions

### `load_and_prepare_data(file_path)`
Loads the dataset from the specified CSV file and performs basic exploratory data analysis.

### `evaluate_model(linreg_model, X_test, y_test)`
Evaluates the linear regression model using various metrics such as R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Explained Variance.

### `visualize_regression_plot_with_metrics(df, linreg_model, X, y, X_test, y_test, feature='FUELCONSUMPTION_COMB', target='CO2EMISSIONS')`
Creates and displays a scatter plot with a linear regression line, including model evaluation metrics.

### `train_linreg_model(df, feature='FUELCONSUMPTION_COMB', target='CO2EMISSIONS')`
Trains a Linear Regression model using a specified feature to predict the target.

## Metrics

- **R² Score**: Indicates the proportion of variance in the target variable explained by the model. Values closer to 1 indicate a better fit.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing error in the same units as the target variable.
- **Explained Variance**: Measures the proportion of variance explained by the model.

## Visualization

The project visualizes the relationship between the specified feature (`FUELCONSUMPTION_COMB`) and the target variable (`CO2EMISSIONS`). The plot includes the regression line and displays the evaluation metrics for easy interpretation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

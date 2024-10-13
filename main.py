import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Constants
FUEL_FILE = "FuelConsumption.csv"

def load_and_prepare_data(file_path):
    """
    Load the dataset and perform basic exploratory data analysis.
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Display basic info about the data
    print("\nDataset Information:")
    print(df.info())
    
    # Display first few rows of the dataset
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    
    # Statistical summary of the dataset
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

def evaluate_model(linreg_model, X_test, y_test):
    """
    Evaluate the linear regression model using various metrics and provide an interpretation.
    """
    y_pred = linreg_model.predict(X_test)
    
    # Calculate different metrics
    r2_score = linreg_model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    explained_variance = explained_variance_score(y_test, y_pred)
    
    # Print out the metrics
    print(f"\nModel Evaluation Metrics:")
    print(f"R² Score: {r2_score:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Explained Variance: {explained_variance:.2f}")
    
    # Interpret the results
    print("\nInterpretation:")
    
    # R² Score Interpretation
    if r2_score == 1:
        print("R²: Perfect prediction (all variability explained).")
    elif r2_score > 0.9:
        print("R²: Excellent model fit (>0.9).")
    elif r2_score > 0.75:
        print("R²: Good model fit (0.75 - 0.9).")
    elif r2_score > 0.5:
        print("R²: Moderate model fit (0.5 - 0.75).")
    else:
        print("R²: Poor model fit (<0.5). Consider improving the model.")
    
    # MAE and RMSE Interpretation
    print(f"MAE and RMSE show average error magnitudes: "
          f"lower values (close to 0) indicate better performance. Current RMSE: {rmse:.2f}")
    
    # Explained Variance Interpretation
    if explained_variance > 0.9:
        print("Explained Variance: Excellent prediction (>0.9).")
    elif explained_variance > 0.75:
        print("Explained Variance: Good prediction (0.75 - 0.9).")
    elif explained_variance > 0.5:
        print("Explained Variance: Moderate prediction (0.5 - 0.75).")
    else:
        print("Explained Variance: Poor prediction (<0.5).")

    return r2_score, mae, mse, rmse

def visualize_regression_plot_with_metrics(df, linreg_model, X, y, X_test, y_test, feature='FUELCONSUMPTION_COMB', target='CO2EMISSIONS'):
    """
    Create and display a scatter plot with a linear regression line, including model evaluation metrics.
    """
    # Predict values for the test set
    y_pred = linreg_model.predict(X_test)
    
    # Calculate metrics
    r2_score, mae, mse, rmse = evaluate_model(linreg_model, X_test, y_test)

    # Plot the scatter plot and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, linreg_model.predict(X), '-r', label=f'Linear Fit: {feature} vs {target}')
    
    # Annotate the plot with evaluation metrics
    plt.text(0.05, 0.85, f"R²: {r2_score:.2f}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.80, f"MAE: {mae:.2f}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.75, f"MSE: {mse:.2f}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.70, f"RMSE: {rmse:.2f}", transform=plt.gca().transAxes, fontsize=12)
    
    plt.title(f"{feature} vs {target} with Regression Line")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_linreg_model(df, feature='FUELCONSUMPTION_COMB', target='CO2EMISSIONS'):
    """
    Train a Linear Regression model using a specified feature to predict the target.
    :param df: DataFrame containing the data
    :param feature: Feature column name (default: FUELCONSUMPTION_COMB)
    :param target: Target column name (default: CO2EMISSIONS)
    :return: Trained Linear Regression model, X and y data, X_test, and y_test
    """
    # Define features (X) and target (y)
    X = df[[feature]]  # Feature
    y = df[[target]]   # Target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Linear Regression model
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    # Evaluate the model
    score = linreg.score(X_test, y_test)
    print(f"\nLinear Regression Model R² Score: {score:.2f}")
    
    return linreg, X, y, X_test, y_test

def main():
    # Step 1: Load and prepare the dataset
    df = load_and_prepare_data(FUEL_FILE)
    
    # Step 2: Train and evaluate the linear regression model
    linreg_model, X, y, X_test, y_test = train_linreg_model(df)
    
    # Step 3: Visualize the dataset and regression results with metrics
    visualize_regression_plot_with_metrics(df, linreg_model, X, y, X_test, y_test)

if __name__ == "__main__":
    main()

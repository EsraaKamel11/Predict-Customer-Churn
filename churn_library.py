# Library docstring
"""
Helper functions for Predicting Customer Churn
Author: Esraa Kamel
Date: Sept. 26th, 2024
"""

# Import libraries
import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.sandbox.stats.multicomp import cv001

sns.set()

# Environment variable to suppress Qt warning
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Ensure the logs directory exists
os.makedirs('./logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def import_data(pth):
    """
    Import dataset and preprocess churn variable.

    Input:
        pth: str - Path to CSV dataset.

    Output:
        dataframe: pandas DataFrame with columns for features and a binary 'Churn' indicator.
    """
    logger.info(f"Starting data import from: {pth}")
    try:
        dataframe = pd.read_csv(pth, index_col=0)
        logger.info(f"Data successfully imported from {pth}.")
    except FileNotFoundError as e:
        logger.error(f"The file at {pth} was not found: {str(e)}")
        raise
    except pd.errors.ParserError as e:
        logger.error(
            f"Error parsing the file at {pth}:{str(e)}. Please check the file format.")
        raise

    if 'Attrition_Flag' not in dataframe.columns or 'CLIENTNUM' not in dataframe.columns:
        logger.error("The required columns are not in the DataFrame.")
        raise ValueError("The required columns are not in the DataFrame.")

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    dataframe.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1, inplace=True)

    # Log data integrity
    logger.info(dataframe.info())
    logger.info(
        f"Data imported with shape: {
            dataframe.shape}. Churned customers: {
            dataframe['Churn'].sum()}")

    return dataframe


def perform_eda(dataframe, output_dir="./images/eda"):
    """
    Perform exploratory data analysis (EDA) on the given DataFrame and save visualizations.

    Input:
        dataframe: pandas DataFrame - The data to analyze.
        output_dir: str - Directory to save EDA images (default: './images/eda').

    Output:
        None
    """
    logger.info("Starting EDA process.")

    # Create a directory for saving EDA images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if DataFrame is empty
    if dataframe.empty:
        logger.warning("DataFrame is empty. Exiting EDA.")
        return

    # Analyze categorical features
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    for cat_column in cat_columns:
        try:
            logger.debug(f"Processing categorical column: {cat_column}")
            plt.figure(figsize=(7, 4))
            ax = dataframe[cat_column].value_counts(normalize=True).plot(
                kind='bar', rot=45, title=f'{cat_column} - % Churn')

            # Annotate the bars with percentages
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2%}',
                            (p.get_x() + p.get_width() / 2.,
                             p.get_height()),
                            ha='center',
                            va='bottom',
                            fontsize=10)

            plt.savefig(
                os.path.join(
                    output_dir,
                    f'{cat_column}.png'),
                bbox_inches='tight')
            plt.close()
            logger.info(f'Saved EDA figure for {cat_column}.')
        except Exception as e:
            logger.error(f'Failed to create EDA plot for {cat_column}: {e}')

    # Numeric analysis (Customer Age, Total Transactions)
    try:
        logger.debug("Generating histogram for Customer Age.")
        plt.figure(figsize=(10, 5))
        dataframe['Customer_Age'].plot(
            kind='hist',
            title='Customer Age Distribution',
            edgecolor='black')
        plt.xlabel('Customer Age')
        plt.savefig(
            os.path.join(
                output_dir,
                'Customer_Age.png'),
            bbox_inches='tight')
        plt.close()
        logger.info('Saved histogram for Customer Age Distribution.')

        logger.debug("Generating histogram for Total Transactions Count.")
        plt.figure(figsize=(10, 5))
        sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Total Transactions Count Distribution')
        plt.savefig(
            os.path.join(
                output_dir,
                'Total_Trans_Ct.png'),
            bbox_inches='tight')
        plt.close()
        logger.info(
            'Saved histogram for Total Transactions Count Distribution.')
    except Exception as e:
        logger.error(f'Failed to create numeric analysis plots: {str(e)}')

        # Correlation matrix (only for numeric columns)
    try:
        logger.debug("Generating correlation matrix heatmap.")
        plt.figure(figsize=(15, 7))
        numeric_df = dataframe.select_dtypes(include=[np.number])
        sns.heatmap(
            numeric_df.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.title('Correlation Matrix')
        plt.savefig(
            os.path.join(
                output_dir,
                'correlation_matrix.png'),
            bbox_inches='tight')
        plt.close()
        logger.info('Saved correlation matrix heatmap.')
    except Exception as e:
        logger.error(f'Failed to create correlation matrix plot: {str(e)}')

    logger.info("EDA completed and visualizations saved.")


def encoder_helper(dataframe, category_lst, response='Churn'):
    """
    Encode categorical columns with the proportion of churn for each category.

    Input:
        dataframe: pandas DataFrame - Input data containing categorical features.
        category_lst: list - List of categorical columns to encode.
        response: str - Target column name (default = 'Churn').

    Output:
        dataframe: pandas DataFrame - DataFrame with encoded features.
    """

    # Check if response column exists
    if response not in dataframe.columns:
        logger.error(f'Response column "{
                     response}" not found in the DataFrame.')
        raise ValueError(
            f'Response column "{response}" not found in the DataFrame.')

    # Check if DataFrame is empty
    if dataframe.empty:
        logger.error("Input DataFrame is empty.")
        raise ValueError("Input DataFrame is empty.")

    # Check for missing categorical columns
    missing_categories = [
        cat for cat in category_lst if cat not in dataframe.columns]
    if missing_categories:
        logger.error(f'Missing categorical columns: {missing_categories}')
        raise ValueError(f'Missing categorical columns in the DataFrame: {
                         missing_categories}')

    for category in category_lst:
        category_groups = dataframe.groupby(category)[response].mean()
        dataframe[f'{category}_{response}'] = dataframe[category].map(
            category_groups)
        logging.info(f'Encoded {category} with churn proportion.')

    # Drop original categorical columns
    dataframe = dataframe.drop(category_lst, axis=1)

    logging.info("Encoding completed successfully.")
    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    """
    Perform feature engineering and split data into train/test sets.

    Input:
        dataframe: pandas DataFrame - Input data for feature engineering.
        response: str - Target column name (default = 'Churn')

    Output:
        X_train, X_test, y_train, y_test: tuple - Train and test datasets split.
    """
    try:
        # Dynamically identify categorical and quantitative columns
        cat_columns = dataframe.select_dtypes(
            include=['object', 'category']).columns.tolist()
        quant_columns = dataframe.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

        # Check if response column exists in the DataFrame
        if response not in dataframe.columns:
            logger.error(
                f"Response column '{response}' not found in the DataFrame.")
            raise ValueError(
                f"Response column '{response}' not found in the DataFrame.")

        # Encode categorical variables and update the DataFrame
        dataframe = encoder_helper(dataframe, cat_columns, response=response)

        # Create an empty DataFrame for features
        X = pd.DataFrame()
        y = dataframe[response]

        # Keep the specified columns, ensuring that they exist in the original
        # DataFrame
        churn_columns = [f'{col}_{response}' for col in cat_columns]
        keep_cols = quant_columns + churn_columns

        # Check if required columns exist in the DataFrame
        missing_cols = [
            col for col in keep_cols if col not in dataframe.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(
                f"Missing columns in the DataFrame: {missing_cols}")

        # Populate the features DataFrame
        X[keep_cols] = dataframe[keep_cols]

        # Handle missing values for quantitative columns
        imputer = SimpleImputer(strategy='mean')
        X[quant_columns] = imputer.fit_transform(X[quant_columns])

        # Check for NaNs after imputation
        if X[quant_columns].isnull().sum().any():
            raise ValueError(
                "Imputation resulted in NaN values in quantitative columns.")

        # Scaling the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert the scaled array back to a DataFrame
        X = pd.DataFrame(X_scaled, columns=keep_cols)

        # Check for NaNs after scaling
        if X.isnull().sum().any():
            raise ValueError(
                "Scaling resulted in NaN values in the feature set.")

        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)

        logger.info(
            f"Feature engineering, imputation, and scaling completed. Features: {keep_cols}")
        logger.info(
            f"X_train shape: {
                X_train.shape}, X_test shape: {
                X_test.shape}, y_train shape: {
                y_train.shape}, y_test shape: {
                    y_test.shape}")

        return X, X_train, X_test, y_train, y_test

    except ValueError as ve:
        logger.error(f"ValueError during feature engineering: {ve}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during feature engineering: {e}")
        raise


def plot_classification_report(
        model_name,
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        output_dir="./images/results"):
    """
    Generate and save classification reports for train and test datasets.

    Input:
        model_name: str - Name of the model.
        y_train, y_test: array-like - Actual response values.
        y_train_preds, y_test_preds: array-like - Predicted values.
        output_dir: str - Directory to save classification report image (default="./images/results").

    Output:
        None
    """
    # Create a directory for saving results if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.rc('figure', figsize=(10, 8))  # Adjust figure size

    try:
        # Generate classification reports for training and testing sets
        plt.text(0.01, 1.25, f'{model_name} Train Classification Report', {
                 'fontsize': 12, 'fontweight': 'bold'}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_train, y_train_preds)), {
                'fontsize': 10}, fontproperties='monospace')

        # Testing report
        plt.text(0.01, 0.55, f'{model_name} Test Classification Report', {
                 'fontsize': 12, 'fontweight': 'bold'}, fontproperties='monospace')
        plt.text(
            0.01, 0.75, str(
                classification_report(
                    y_test, y_test_preds)), {
                'fontsize': 10}, fontproperties='monospace')

        plt.axis('off')
        plt.savefig(
            os.path.join(
                output_dir,
                f'Classification_report_{model_name}.png'),
            bbox_inches='tight')
        plt.close()

        logging.info(f'Classification report for {
                     model_name} saved successfully.')

    except Exception as e:
        logging.error(
            f'Error generating classification report for {model_name}: {e}')
        raise RuntimeError(
            f'Failed to generate classification report for {model_name}: {e}')


def validate_input_shapes(y_true, y_preds, model_name):
    """
    Validate that true and predicted values have the same shape.

    Input:
        y_true: array-like - Actual values.
        y_preds: array-like - Predicted values.
        model_name: str - Name of the model.

    Raises:
        ValueError if the shapes do not match.
    """
    if y_preds.shape != y_true.shape:
        raise ValueError(f'Shape mismatch for {model_name}: predictions and true values must have the same shape. '
                         f'Predictions shape: {y_preds.shape}, True values shape: {y_true.shape}')


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    """
    Generate and save classification reports for Logistic Regression and Random Forest models.

    Input:
        y_train: pandas Series or numpy array - Actual response values for the training set.
        y_test: pandas Series or numpy array - Actual response values for the test set.
        y_train_preds_lr: pandas Series or numpy array - Predictions from Logistic Regression on the training set.
        y_train_preds_rf: pandas Series or numpy array - Predictions from Random Forest on the training set.
        y_test_preds_lr: pandas Series or numpy array - Predictions from Logistic Regression on the test set.
        y_test_preds_rf: pandas Series or numpy array - Predictions from Random Forest on the test set.

    Output:
        None
    """
    # Dictionary for models and their respective predictions
    models = {
        'Logistic Regression': (y_train_preds_lr, y_test_preds_lr),
        'Random Forest': (y_train_preds_rf, y_test_preds_rf)
    }

    # Validate shapes and generate reports
    try:
        for model_name, (train_preds, test_preds) in models.items():
            validate_input_shapes(y_train, train_preds, model_name)
            validate_input_shapes(y_test, test_preds, model_name)

            # Generate classification reports for train and test
            plot_classification_report(
                model_name, y_train, y_test, train_preds, test_preds)

        logging.info(
            'Classification reports generated successfully for all models.')

    except Exception as e:
        logging.error(f'Error generating classification reports: {e}')
        print(
            f"An error occurred while generating classification reports: {e}")


def feature_importance_plot(model, X_data, model_name, figsize=(20, 5)):
    """
    Create and save a feature importance plot.

    Input:
        X: Original dataframe.
        model: Trained model with feature_importances_ attribute (e.g., RandomForestClassifier).
        X_data: pandas DataFrame - Features used for training the model.
        model_name: str - Name of the model (e.g., 'Random Forest').
        output_pth: str - Directory to save the plot.
        figsize: tuple - Size of the figure (width, height) (default=(20, 5)).
        color: str - Color of the bars in the plot (default='skyblue').

    Output:
        str - Path of the saved feature importance plot image.
    """

    # Validate input types
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(
            "The model must have 'feature_importances_' attribute.")
    if not isinstance(X_data, pd.DataFrame):
        raise TypeError("X_data must be a pandas DataFrame.")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    # Create the output directory if it doesn't exist
    os.makedirs('./images/results', exist_ok=True)

    try:
        # Extract feature importances and sort them
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names to match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=figsize)
        plt.title(f"Feature Importance for {model_name}", fontsize=16)
        plt.ylabel('Importance', fontsize=12)

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90, fontsize=10)

        # Save the figure
        output_file = os.path.join(
            './images/results',
            f'feature_importance_{model_name}.png')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logging.info(f'Feature importance plot for {
                     model_name} saved successfully at {output_file}.')

    except Exception as e:
        logging.error(
            f'Error generating feature importance plot for {model_name}: {e}')
        raise RuntimeError(
            f'Failed to generate feature importance plot for {model_name}: {e}')


def confusion_matrix_plot(
        model,
        model_name,
        X_test,
        y_test,
        output_dir="./images/results"):
    """
    Plot and save the confusion matrix for a given model.

    Input:
        model: Trained model (should be fitted).
        model_name: str - Name of the model (e.g., 'Logistic Regression').
        X_test: pandas DataFrame - Features of the test dataset.
        y_test: pandas Series - Actual response values of the test dataset.
        output_dir: str - Directory to save the confusion matrix plot (default="./images/results").

    Output:
        str - Path of the saved confusion matrix plot image.
    """
    # Check if the model is fitted
    if not hasattr(model, 'predict'):
        raise ValueError(
            f"The model is not fitted or does not have a 'predict' method: {model_name}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Predict the values for the test set
        y_pred = model.predict(X_test)

        # Create figure and axes explicitly
        fig, ax = plt.subplots(figsize=(10, 7))

        # Generate the confusion matrix display
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=['Not Churned', 'Churned'],
            cmap=plt.cm.Blues,
            ax=ax  # Use the explicit axes for plotting
        )

        ax.set_title(f'{model_name} Confusion Matrix', fontsize=16)

        # Save the figure
        output_file = os.path.join(
            output_dir, f'{model_name}_Confusion_Matrix.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

        logging.info(f'Confusion matrix for {
                     model_name} saved successfully at {output_file}.')
        return output_file  # Return the path of the saved plot

    except Exception as e:
        logging.error(
            f'Error generating confusion matrix for {model_name}: {e}')
        raise RuntimeError(
            f'Failed to generate confusion matrix for {model_name}: {e}')


def plot_roc_curve(
        model,
        X_test,
        y_test,
        model_name,
        output_dir="./images/results"):
    """
    Plot and save the ROC curve for a given model.

    Input:
        model: Trained model (should be fitted).
        X_test: pandas DataFrame - Test feature set.
        y_test: pandas Series - True labels for the test set.
        model_name: str - Name of the model (e.g., 'Logistic Regression').
        output_dir: str - Directory to save the ROC curve plot (default="./images/results").

    Output:
        str - Path of the saved ROC curve plot image.
    """
    # Check if the model is fitted and supports probability predictions
    if not hasattr(model, 'predict_proba'):
        raise ValueError(
            f"The model is not fitted or does not support probability predictions: {model_name}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create figure and axes explicitly
        fig, ax = plt.subplots(figsize=(10, 5))

        # Generate and display the ROC curve
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

        # Customize plot title and labels
        ax.set_title(f'{model_name} ROC Curve', fontsize=16)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.grid()

        # Save the figure
        output_file = os.path.join(output_dir, f'{model_name}_ROC_Curve.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

        logging.info(
            f'ROC curve for {model_name} saved successfully at {output_file}.')
        return output_file  # Return the path of the saved plot

    except Exception as e:
        logging.error(f'Error generating ROC curve for {model_name}: {e}')
        raise RuntimeError(
            f'Failed to generate ROC curve for {model_name}: {e}')


def train_models(X, X_train, X_test, y_train, y_test):
    """
    Train models, generate evaluation metrics, and save results.

    Input:
        X_train: pandas DataFrame - training feature set
        X_test: pandas DataFrame - testing feature set
        y_train: pandas Series - training response variable
        y_test: pandas Series - testing response variable

    Output:
        None
    """
    # Ensure the output directory exists
    output_dir = './images/results'
    os.makedirs(output_dir, exist_ok=True)

    # Logistic Regression Model
    # Logistic Regression Model
    try:
        logging.info("Training Logistic Regression model...")
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        lrc.fit(X_train, y_train)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        logging.info(
            "Logistic Regression model training completed successfully.")

        # Generate plots for Logistic Regression
        confusion_matrix_plot(lrc, 'Logistic Regression', X_test, y_test)
        plot_roc_curve(lrc, X_test, y_test, 'Logistic Regression')
        lrc_report = plot_classification_report(
            'Logistic Regression',
            y_train,
            y_test,
            y_train_preds_lr,
            y_test_preds_lr)
        logging.info(
            "Logistic Regression Classification Report:\n%s",
            lrc_report)

    except Exception as e:
        logging.error(f"Error training Logistic Regression: {e}")
        raise RuntimeError(f"Error training Logistic Regression: {e}")

    # Random Forest Model
    try:
        logging.info("Training Random Forest model...")
        rfc = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        logging.info("Random Forest model training completed successfully.")

        # Generate plots for Random Forest
        confusion_matrix_plot(
            cv_rfc.best_estimator_,
            'Random Forest',
            X_test,
            y_test)
        plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, 'Random Forest')
        rfc_report = plot_classification_report(
            'Random Forest', y_train, y_test, y_train_preds_rf, y_test_preds_rf)
        logging.info("Random Forest Classification Report:\n%s", rfc_report)

        # Feature Importance Plot for Random Forest
        feature_importance_plot(cv_rfc.best_estimator_, X, 'Random Forest')

    except Exception as e:
        logging.error(f"Error training Random Forest: {e}")
        raise RuntimeError(f"Error training Random Forest: {e}")

    # Save the models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    # Set the data path
    data_path = 'data/bank_data.csv'

    # Import and preprocess data
    df = import_data(data_path)
    # Perform exploratory data analysis
    perform_eda(df)

    # Split the dataset for training and testing
    X, X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    # Train models and generate reports/plots
    train_models(X, X_train, X_test, y_train, y_test)

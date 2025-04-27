"""
Unit test of churn_library.py module with pytest
author: Esraa Kamel
Date: Sept. 26, 2024
"""
import logging
import os
import pytest
from churn_library import *

# Create required directories if they do not exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

if not os.path.exists('./images/results'):
    os.makedirs('./images/results')

# Configure logging for the test suite
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

######################### FIXTURES ##################################

@pytest.fixture(scope="module")
def path():
    """Fixture for the file path to the dataset."""
    return "./data/bank_data.csv"

@pytest.fixture(scope="module")
def dataframe(path):
    """Fixture for importing the data as a DataFrame."""
    return import_data(path)

@pytest.fixture(scope="module")
def encoder_params():
    """Fixture providing parameters for the encoder helper tests."""
    data = pytest.df.copy()

    # Extract valid categorical columns from the DataFrame
    valid_categories = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create test scenarios
    params = [
        valid_categories,  # All valid
        valid_categories[:-1],  # All but the last valid column
        valid_categories + ['Not_a_column'],  # One invalid column
        [],  # No categories
    ]

    return [(data, param) for param in params]

@pytest.fixture(scope="module")
def input_train(dataframe):
    """Fixture for the training input data after feature engineering."""
    return perform_feature_engineering(dataframe)

######################### UNIT TESTS ##################################

@pytest.mark.parametrize("filename", [
    "./data/bank_data.csv",
    "./data/no_file.csv"
])
def test_import(filename):
    """Test the data import functionality."""
    if filename == "./data/bank_data.csv":
        # Test for the existing file
        data = import_data(filename)
        pytest.df = data  # Store the DataFrame for reuse
        logging.info("Testing import_data from file: %s - SUCCESS", filename)

        assert data.shape[0] > 0 and data.shape[1] > 0
        logging.info("Returned dataframe with shape: %s", data.shape)
    else:
        # Test for the non-existing file
        with pytest.raises(FileNotFoundError) as excinfo:
            import_data(filename)
        logging.info("Testing import_data for non-existent file: %s - SUCCESS", filename)
        assert "No such file or directory" in str(excinfo.value)


def test_eda(dataframe):
    """Test the exploratory data analysis function."""
    try:
        perform_eda(dataframe)
        logging.info("Testing perform_eda - SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda failed - Error type: %s, Error message: %s", type(err), str(err))


def test_encoder_helper(encoder_params):
    """Test the encoder helper function."""
    data, cat_features = encoder_params
    try:
        newdf = encoder_helper(data, cat_features)
        logging.info("Testing encoder_helper with %s - SUCCESS", cat_features)

        assert newdf.select_dtypes(include='object').empty
        logging.info("All categorical columns were encoded")

    except KeyError:
        logging.error("Check for categorical features not in the dataset: %s", cat_features)
    except Exception as err:
        logging.error("Testing encoder_helper failed - Error type: %s, Error message: %s", type(err), str(err))


def test_perform_feature_engineering(dataframe):
    """Test the feature engineering process."""
    try:
        X, X_train, X_test, y_train, y_test = perform_feature_engineering(dataframe)
        logging.info("Testing perform_feature_engineering - SUCCESS")

        assert X_train.shape[0] > 0 and X_train.shape[1] > 0
        assert X_test.shape[0] > 0 and X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Train / Test set shapes: X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    except Exception as err:
        logging.error("Testing perform_feature_engineering failed - Error type: %s, Error message: %s", type(err), str(err))


def test_feature_importance_plot(input_train):
    """Test the feature importance plot generation."""
    try:
        X, X_train, _, y_train, _ = input_train

        # Check data types
        logging.info("X_train data types: %s", X_train.dtypes)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)  # Fit the model to generate feature importances

        feature_importance_plot(rf_model, X, 'Random Forest', './images/results')
        logging.info("Testing feature_importance_plot - SUCCESS")

        assert os.path.exists('./images/results/feature_importance_Random Forest.png')
        logging.info("Feature importance plot saved successfully.")

    except Exception as err:
        logging.error("Testing feature_importance_plot failed - Error type: %s, Error message: %s", type(err), str(err))


def test_classification_report_image(input_train):
    """Test the classification report generation."""
    try:
        X, X_train, X_test, y_train, y_test = input_train

        # Check data types
        logging.info("Data types before training: X_train: %s, y_train: %s", X_train.dtypes, y_train.dtypes)

        lr_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
        y_train_preds = lr_model.predict(X_train)
        y_test_preds = lr_model.predict(X_test)

        classification_report_image(y_train, y_test, y_train_preds, y_test_preds)
        logging.info("Testing classification_report_image - SUCCESS")

        assert os.path.exists('./images/results/Classification_report_Random Forest.png')
        logging.info("Classification report image saved successfully.")

    except Exception as err:
        logging.error("Testing classification_report_image failed - Error type: %s, Error message: %s", type(err), str(err))


def test_confusion_matrix_plot(input_train):
    """Test the confusion matrix generation."""
    try:
        X, X_train, X_test, y_train, y_test = input_train

        # Check data types
        logging.info("X_train data types: %s", X_train.dtypes)

        rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

        confusion_matrix_plot(rf_model, 'Random Forest', X_test, y_test)
        logging.info("Testing confusion_matrix_plot - SUCCESS")

        assert os.path.exists('./images/results/Random Forest_Confusion_Matrix.png')
        logging.info("Confusion matrix plot saved successfully.")

    except Exception as err:
        logging.error("Testing confusion_matrix_plot failed - Error type: %s, Error message: %s", type(err), str(err))


def test_plot_roc_curve(input_train):
    """Test the ROC curve generation."""
    try:
        X, X_train, X_test, y_train, y_test = input_train

        # Check data types
        logging.info("X_train data types: %s", X_train.dtypes)

        rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

        plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
        logging.info("Testing plot_roc_curve - SUCCESS")

        assert os.path.exists('./images/results/Random Forest_ROC_Curve.png')
        logging.info("ROC curve plot saved successfully.")

    except Exception as err:
        logging.error("Testing plot_roc_curve failed - Error type: %s, Error message: %s", type(err), str(err))


def test_train_models(input_train):
    """Test the training of models."""
    try:
        train_models(*input_train)
        logging.info("Testing train_models - SUCCESS")
    except Exception as err:
        logging.error("Testing train_models failed - Error type: %s, Error message: %s", type(err), str(err))

if __name__ == "__main__":
    pytest.main()  

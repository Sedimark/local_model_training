import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tuner import Tuner  
from model import XGBoostModel
from sklearn.preprocessing import StandardScaler

# Title and Description
st.title("XGBoost Model Trainer and Tuner")
st.write("This application allows you to upload a dataset, tune XGBoost parameters, train the model, and evaluate its performance.")

# Step 1: Data Upload
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(data.head(10))

    # Allow user to select target column
    target_column = st.selectbox("Select the target column", data.columns)

    # Step 2: Parameter Selection
    st.header("2. Select Parameters to Tune")

    # XGBoost parameters for tuning
    xgboost_parameters = [
        'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 
        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'early_stopping_rounds'
    ]

    selected_params = st.multiselect(
        "Select parameters to tune:", xgboost_parameters, default=['n_estimators', 'learning_rate']
    )

    st.write("### Selected Parameters:")
    st.write(selected_params)

    # Step 3: Train-Test Split
    st.header("3. Train-Test-Validation Split")
    st.write("Split your dataset into training, testing, and validation sets.")

    train_size = st.slider("Train Size (%)", 50, 80, 70, step=5) / 100
    test_size = st.slider("Test Size (%)", 10, 20, 15, step=5) / 100
    validation_size = 1-train_size-test_size

    test_size_temp = test_size / (1-train_size)

    st.write("The validation set is used for hyperparameter tuning and is created from the remaining portion of the dataset after splitting it into training and testing sets.")

    if st.button("Tune & Train Model"):
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column].values.reshape(-1, 1)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(train_size, test_size, validation_size, test_size_temp)

        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()

        # X_train_scaled = scaler_X.fit_transform(X_train)
        # X_val_scaled = scaler_X.transform(X_val)
        # X_test_scaled = scaler_X.transform(X_test)

        # y_train_scaled = scaler_y.fit_transform(y_train)
        # y_val_scaled = scaler_y.transform(y_val)
        # y_test_scaled = scaler_y.transform(y_test)
        
        # # Tune parameters using the external function
        # st.write("Tuning parameters...")
        # tuner = Tuner(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'XGBoostRegressor')
        # best_params = tuner.tune(n_trials = 100)

        st.write("Tuning parameters...")
        tuner = Tuner(X_train, y_train, X_val, y_val, 'XGBoostRegressor')
        best_params = tuner.tune(n_trials = 100)

        st.write("### Best Parameters:")
        st.json(best_params)

        # # Train model with best parameters
        # best_model = XGBoostModel(**best_params)
        # best_model.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

        best_model = XGBoostModel(**best_params)
        best_model.train(X_train, y_train, X_val, y_val)

        # Step 4: Model Evaluation
        st.header("4. Model Evaluation")

        # Predictions and Metrics
        # y_pred = best_model.predict(X_test_scaled)

        # predictions= scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        # y_test_actual = scaler_y.inverse_transform(y_test_scaled)

        y_pred = best_model.predict(X_test)

        # Metrics
        # Calculate metrics
        # mae = mean_absolute_error(y_test_actual, predictions)
        # mse = mean_squared_error(y_test_actual, predictions)
        # rmse = np.sqrt(mse)  # RMSE is the square root of MSE
        # mape = mean_absolute_percentage_error(y_test_actual, predictions)
        # r2 = r2_score(y_test_actual, predictions)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # RMSE is the square root of MSE
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        metrics_values = [mae, mse, rmse, mape, r2]

        # Display the metrics

        st.write("### Metrics")
        comparison_metrics = pd.DataFrame({"Metrics": metrics, "Values": metrics_values})
        st.write(comparison_metrics.head(10))

        # st.markdown(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        # st.markdown(f"**Mean Squared Error (MSE):** {mse:.4f}")
        # st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        # st.markdown(f"**Mean Absolute Percentage Error (MAPE):** {mape:.4f}")
        # st.markdown(f"**R-Squared (R²):** {r2:.4f}")


        # # Ensure both arrays are 1-dimensional
        # y_test_actual = y_test_actual.flatten()
        # predictions= predictions.flatten()

        # st.write("### Predictions vs True Values")
        # comparison = pd.DataFrame({"True Values": y_test_actual.tolist(), "Predictions": predictions.tolist()})
        # st.write(comparison.head(10))

        st.write("### Predictions vs True Values")
        comparison_pred_true = pd.DataFrame({"True Values": y_test.tolist(), "Predictions": y_pred.tolist()})
        st.write(comparison_pred_true.head(10))

        st.write("### Feature Importances")
        feature_importances = best_model.model.feature_importances_
        st.bar_chart(pd.DataFrame({"Features": X.columns, "Importance": feature_importances}).set_index("Features"))


        st.write("### Analysis")

        # # Line Plot: Mean Energy consumption
        # fig1, ax1 = plt.subplots(figsize=(10, 6))
        # ax1.plot(predictions, label="Predictions", linewidth=3, color='red')
        # ax1.plot(y_test_actual, label="True Values", linewidth=3, color='green')
        # ax1.set_xlabel('Instances', fontsize=15)
        # ax1.set_ylabel('Mean daily consumption [kWh]', fontsize=15)
        # ax1.set_title('Mean daily consumption for each instance', fontsize=17)
        # ax1.legend()
        # st.pyplot(fig1)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(y_pred, label="Predictions", linewidth=3, color='red')
        ax1.plot(y_test, label="True Values", linewidth=3, color='green')
        ax1.set_xlabel('Instances', fontsize=15)
        ax1.set_ylabel('Mean daily consumption [kWh]', fontsize=15)
        ax1.set_title('Mean daily consumption for each instance', fontsize=17)
        ax1.legend()
        st.pyplot(fig1)


        # # Scatter Plot: Predictions vs Actual
        # fig2, ax2 = plt.subplots(figsize=(10, 6))
        # ax2.scatter(y_test_actual, predictions, alpha=0.6,label='Predictions vs Actual')
        # ax2.plot([y_test_actual.min(), y_test_actual.max()], 
        #         [y_test_actual.min(), y_test_actual.max()], 
        #         color='r', linestyle='--', label='y = x')
        # ax2.set_xlabel('Actual Values [kWh]', fontsize=15)
        # ax2.set_ylabel('Predicted Values [kWh]', fontsize=15)
        # ax2.set_title('Scatter Plot: Predictions vs Actual with y = x Line', fontsize=17)
        # ax2.legend()
        # st.pyplot(fig2)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_test, y_pred, alpha=0.6,label='Predictions vs Actual')
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                color='r', linestyle='--', label='y = x')
        ax2.set_xlabel('Actual Values [kWh]', fontsize=15)
        ax2.set_ylabel('Predicted Values [kWh]', fontsize=15)
        ax2.set_title('Scatter Plot: Predictions vs Actual with y = x Line', fontsize=17)
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.write("Click 'Tune & Train Model' to start tuning and training.")
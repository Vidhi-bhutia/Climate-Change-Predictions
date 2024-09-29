import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Custom CSS styling for headings
def set_custom_css():
    st.markdown("""
        <style>
        h1 {
            color: #FFA500;  /* Orange - visible in both light and dark mode */
        }
        h2 {
            color: #00BFFF;  /* DeepSkyBlue - visible in both modes */
        }
        h3 {
            color: #ADFF2F;  /* GreenYellow - visible in both modes */
        }
        h4 {
            color: #FF4500;  /* OrangeRed for the final result */
        }
        </style>
        """, unsafe_allow_html=True)

# Call custom CSS function
set_custom_css()

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('climate_change_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    X = df.drop(['Date', 'Location', 'Country', 'Temperature'], axis=1)
    y = df['Temperature']
    return X, y

X, y = load_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load models
rf_model = joblib.load('rf_model_optimized.joblib')
gb_model = joblib.load('gb_model_optimized.joblib')
nn_model = joblib.load('nn_model_optimized.joblib')
rfi_model = joblib.load('rf_model_initial.joblib')
gbi_model = joblib.load('gb_model_initial.joblib')
nni_model = joblib.load('nn_model_initial.joblib')

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test_scaled)
y_pred_rfi = rfi_model.predict(X_test)
y_pred_gbi = gbi_model.predict(X_test)
y_pred_nni = nni_model.predict(X_test_scaled)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, mape

metrics_rf = calculate_metrics(y_test, y_pred_rf)
metrics_gb = calculate_metrics(y_test, y_pred_gb)
metrics_nn = calculate_metrics(y_test, y_pred_nn)
metrics_rfi = calculate_metrics(y_test, y_pred_rfi)
metrics_gbi = calculate_metrics(y_test, y_pred_gbi)
metrics_nni = calculate_metrics(y_test, y_pred_nni)

# Streamlit app with color customized headings
st.markdown('<h1>Climate Change Model Comparison</h1>', unsafe_allow_html=True)

# Individual model plots
def plot_model_results(y_true, y_pred, model_name):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot
    axs[0, 0].scatter(y_true, y_pred)
    axs[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axs[0, 0].set_xlabel('Actual Temperature')
    axs[0, 0].set_ylabel('Predicted Temperature')
    axs[0, 0].set_title('Actual vs Predicted Temperature')
    
    # Residual plot
    residuals = y_true - y_pred
    axs[0, 1].scatter(y_pred, residuals)
    axs[0, 1].axhline(y=0, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Predicted Temperature')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].set_title('Residual Plot')
    
    # Distribution plot
    sns.kdeplot(y_true, ax=axs[1, 0], label='Actual', shade=True)
    sns.kdeplot(y_pred, ax=axs[1, 0], label='Predicted', shade=True)
    axs[1, 0].set_xlabel('Temperature')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].set_title('Distribution of Actual vs Predicted Temperature')
    axs[1, 0].legend()
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('<h2>Individual Model Results</h2>', unsafe_allow_html=True)

st.markdown('<h3>Random Forest Model</h3>', unsafe_allow_html=True)
plot_model_results(y_test, y_pred_rf, 'Random Forest')

st.markdown('<h3>Gradient Boosting Model</h3>', unsafe_allow_html=True)
plot_model_results(y_test, y_pred_gb, 'Gradient Boosting')

st.markdown('<h3>Neural Network Model</h3>', unsafe_allow_html=True)
plot_model_results(y_test, y_pred_nn, 'Neural Network')

# Comparison plot
st.markdown('<h2>Model Comparison</h2>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest')
ax.scatter(y_test, y_pred_gb, alpha=0.5, label='Gradient Boosting')
ax.scatter(y_test, y_pred_nn, alpha=0.5, label='Neural Network')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Temperature')
ax.set_ylabel('Predicted Temperature')
ax.set_title('Actual vs Predicted Temperature - All Models')
ax.legend()
st.pyplot(fig)

# Error comparison tables
st.markdown('<h2>Error Metrics Comparison</h2>', unsafe_allow_html=True)

initial_errors = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
    'MSE': [metrics_rfi[0], metrics_gbi[0], metrics_nni[0]],
    'RMSE': [metrics_rfi[1], metrics_gbi[1], metrics_nni[1]],
    'MAE': [metrics_rfi[2], metrics_gbi[2], metrics_nni[2]],
    'MAPE': [metrics_rfi[3], metrics_gbi[3], metrics_nni[3]]
}

st.markdown('<h3>Initial Errors</h3>', unsafe_allow_html=True)
st.table(pd.DataFrame(initial_errors))

optimized_errors = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
    'MSE': [metrics_rf[0], metrics_gb[0], metrics_nn[0]],
    'RMSE': [metrics_rf[1], metrics_gb[1], metrics_nn[1]],
    'MAE': [metrics_rf[2], metrics_gb[2], metrics_nn[2]],
    'MAPE': [metrics_rf[3], metrics_gb[3], metrics_nn[3]]
}

st.markdown('<h3>Optimized Errors</h3>', unsafe_allow_html=True)
st.table(pd.DataFrame(optimized_errors))

# Final result statement
st.markdown('<h2>Final Result</h2>', unsafe_allow_html=True)

best_model = min(['Random Forest', 'Gradient Boosting', 'Neural Network'], 
                 key=lambda x: optimized_errors['MSE'][optimized_errors['Model'].index(x)])

st.markdown(f'<h4>Based on the lowest Mean Squared Error (MSE), the most accurate model is the {best_model} model.</h4>', unsafe_allow_html=True)

st.write("""
Note: While these models provide insights into the relationships between various climate factors and temperature, they should be used cautiously for predictions. Climate systems are complex and can be influenced by many factors not captured in this dataset. Always consult with climate scientists and experts when making decisions based on climate predictions.
""")

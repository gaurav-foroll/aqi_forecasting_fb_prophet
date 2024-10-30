import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# Load the dataset
air_quality = pd.read_csv('air_quality.csv')  # Make sure this file is in the same directory
air_quality['Date & Time'] = pd.to_datetime(air_quality['Date & Time'])

# Prepare the data for Prophet
data = pd.DataFrame()
data['ds'] = air_quality['Date & Time']
data['y'] = air_quality['AQI']

# Train the model
model = Prophet()
model.fit(data)

# Forecasting
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Load test data
test_data = pd.read_csv('test_data.csv')  # Make sure this file is in the same directory
test_data['Date & Time'] = pd.to_datetime(test_data['Date & Time'], format='%m/%d/%y %I:%M %p', errors='coerce')

# Extract the date and ensure it's datetime
daily_test_data = test_data.groupby(test_data['Date & Time'].dt.date)['AQI'].mean().reset_index()
daily_test_data.columns = ['ds', 'y']

# Convert 'ds' column in daily_test_data to datetime
daily_test_data['ds'] = pd.to_datetime(daily_test_data['ds'])

# Ensure 'ds' in forecast is also in datetime
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Prepare the comparison DataFrame
comparison = pd.merge(forecast[['ds', 'yhat']], daily_test_data, on='ds')
comparison.rename(columns={'yhat': 'y_predicted', 'y': 'y_actual'}, inplace=True)

# Calculate evaluation metrics
mae = mean_absolute_error(comparison['y_actual'], comparison['y_predicted'])
rmse = mean_squared_error(comparison['y_actual'], comparison['y_predicted'], squared=False)

# Streamlit layout
st.title("Air Quality Index Prediction")
st.write("### Mean Absolute Error:", mae)
st.write("### Root Mean Square Error:", rmse)

# Plot actual vs predicted
fig1 = px.line(comparison, x='ds', y=['y_actual', 'y_predicted'], labels={'value': 'AQI', 'variable': 'Legend'}, title='Actual vs Predicted AQI')
st.plotly_chart(fig1)

# Plot residuals
comparison['residuals'] = comparison['y_actual'] - comparison['y_predicted']
fig2 = px.scatter(comparison, x='ds', y='residuals', title='Residuals of AQI Predictions')
fig2.add_shape(type='line', x0=comparison['ds'].min(), y0=0, x1=comparison['ds'].max(), y1=0, line=dict(color='red', dash='dash'))
st.plotly_chart(fig2)

# Additional plot: AQI over time (original data)
fig3 = px.line(air_quality, x='Date & Time', y='AQI', title='Air Quality Index Over Time')
st.plotly_chart(fig3)

# Additional plot: Histogram of AQI values
fig4 = px.histogram(air_quality, x='AQI', title='Distribution of AQI Values')
st.plotly_chart(fig4)

# Plot Prophet components
st.subheader("Prophet Model Components")
components_fig = model.plot_components(forecast)
st.pyplot(components_fig)

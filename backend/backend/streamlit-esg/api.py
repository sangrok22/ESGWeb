import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load("best_esg_prediction_model.pkl")

# Function to create lag features
def create_lag_features(data, target_column, lag_days):
    """Create lag features for time series data."""
    df = data.copy()
    for lag in range(1, lag_days + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

# Function to generate future dates
def generate_future_dates(last_date, num_days):
    """Generate future dates for prediction."""
    future_dates = []
    for i in range(1, num_days + 1):
        future_dates.append(last_date + timedelta(days=i))
    return future_dates

# Function to forecast the next 3 days based on the model and average data
def forecast_next_days(model, avg_data, num_days=3):
    """Forecast the next specified number of days using the average data."""
    predictions = []
    current_input = avg_data.copy()

    for _ in range(num_days):
        # Reshape for prediction
        prediction = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(prediction)

        # Update the input data for the next prediction
        current_input = np.roll(current_input, -1)
        current_input[-1] = prediction

    return predictions

# Streamlit app setup
st.title("ESG Score Prediction")

# Display a message explaining what ESG is
st.write("""
    **Environmental, Social, and Governance (ESG) Score**:
    ESG scores evaluate the sustainability and ethical impact of a company or investment based on three key criteria:
    - **Environmental (E)**: Measures a company’s efforts in managing its environmental impact, such as reducing carbon emissions or conserving resources.
    - **Social (S)**: Evaluates how a company manages relationships with employees, suppliers, customers, and communities.
    - **Governance (G)**: Assesses a company’s leadership, executive pay, audits, and shareholder rights.

    A higher ESG score typically indicates that a company has stronger sustainability practices and ethical standards.
""")

# Allow the user to upload a CSV or Excel file
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            input_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            input_data = pd.read_excel(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:", input_data)

        # Check if the DataFrame has the required columns
        required_columns = [
            "Date", "Industry", "Carbon_Emissions", "Governance_Score",
            "Social_Score", "Environmental_Score", "ESG_Score"
        ]
        
        if all(col in input_data.columns for col in required_columns):
            st.success("Data uploaded successfully. Let's start predicting ESG scores for the next 3 days.")

            # Sort the data by Date and create lag features (as in the training process)
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            input_data = input_data.sort_values('Date')
            
            # Create lag features (e.g., ESG_Score_lag_1, ESG_Score_lag_2, etc.)
            target_column = 'ESG_Score'
            lag_days = 30  # Use the same lag days used during model training
            input_data = create_lag_features(input_data, target_column, lag_days)

            # Drop rows with NaN values due to lag creation
            input_data = input_data.dropna()

            # Ensure that the required feature columns are present
            feature_columns = [col for col in input_data.columns if col not in ['Date', 'Industry', 'ESG_Score']]
            input_data_features = input_data[feature_columns]

            # Calculate the average of all rows for each feature
            avg_data = input_data_features.mean().values

            # Forecast the next 3 days using the model and the average data
            future_predictions = forecast_next_days(model, avg_data, num_days=3)

            # Generate dates for predictions
            future_dates = generate_future_dates(input_data['Date'].iloc[-1], 3)

            # Display the predictions
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_ESG_Score': future_predictions
            })

            st.write("Predictions for the next 3 days:", prediction_df)

            # Add date picker to select the range for visualization
            st.write("### Select a date range to visualize the data:")

            start_date = st.date_input("Start Date", input_data['Date'].min())
            end_date = st.date_input("End Date", input_data['Date'].max())

            # Filter data based on the selected range
            filtered_data = input_data[(input_data['Date'] >= pd.to_datetime(start_date)) & (input_data['Date'] <= pd.to_datetime(end_date))]

            # Combine historical ESG scores with predicted values for comparison
            historical_data = filtered_data[['Date', 'ESG_Score']]  # Filter historical data within the range
            predicted_data = pd.DataFrame({
                'Date': future_dates,
                'ESG_Score': future_predictions
            })

            # Use pd.concat() to combine historical data and predicted data
            comparison_data = pd.concat([historical_data, predicted_data], ignore_index=True)

            # Plotting the ESG Scores (Actual vs Predicted)
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=comparison_data, x='Date', y='ESG_Score', marker='o', label='Actual & Predicted ESG Scores')
            plt.title(f"ESG Scores: Historical vs Predicted for the Next 3 Days ({start_date} to {end_date})")
            plt.xlabel('Date')
            plt.ylabel('ESG Score')
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot()

            # Optionally, allow the user to download the results as a CSV
            st.download_button(
                label="Download Prediction Results",
                data=prediction_df.to_csv(index=False),
                file_name="predicted_esg_scores.csv",
                mime="text/csv"
            )
        else:
            st.error("Uploaded file is missing one or more required columns. Please ensure the following columns are present: 'Date', 'Industry', 'Carbon_Emissions', 'Governance_Score', 'Social_Score', 'Environmental_Score', 'ESG_Score'.")
            st.info("Each column should represent a feature relevant to calculating ESG scores, such as environmental impact (e.g., carbon emissions), governance practices, and social considerations.")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.warning("Please ensure the data is in the correct format. It should be a table with the necessary columns for accurate predictions.")

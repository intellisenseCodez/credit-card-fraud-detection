import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import os
from datetime import datetime

# --- 1. SETUP AND MODEL LOADING ---

# Set page configuration for a more professional look
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Function to load the trained model
# Using st.cache_resource to load the model only once, improving performance
@st.cache_resource
def load_model():
    """Loads the pre-trained LightGBM model from a pickle file."""
    try:
        with open('lgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'lgb_model.pkl' not found. Please ensure it's in the same directory as this script.")
        st.stop()

# Load the model
model = load_model()

# --- 2. PREPROCESSING AND HELPER FUNCTIONS ---

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on Earth (in kilometers)
    using the Haversine formula. This matches the function from your notebook.
    """
    R = 6371  # Earth radius in Km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Function to get mappings for categorical features from the original dataset
# This is crucial to mimic the LabelEncoder used during training
@st.cache_data
def get_mappings(data_path='fraudTrain.csv'):
    """
    Loads the original training data to create mappings for categorical variables.
    This replicates the LabelEncoder transformation from the notebook.
    """
    if not os.path.exists(data_path):
        st.error(f"Error: The data file '{data_path}' was not found.")
        st.error("Please download 'fraudTrain.csv' and place it in the same directory as this app.")
        st.stop()

    df = pd.read_csv(data_path)
    mappings = {}
    cat_cols = ['category', 'gender', 'city', 'state', 'job']
    for col in cat_cols:
        # Sort unique values to ensure consistent encoding
        unique_vals = sorted(df[col].unique().tolist())
        mappings[col] = {val: i for i, val in enumerate(unique_vals)}
        # Store the list of unique values for dropdown menus
        mappings[col + '_list'] = unique_vals
    return mappings

# Load the mappings needed for the UI and preprocessing
mappings = get_mappings()


# --- 3. STREAMLIT USER INTERFACE ---

st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This application uses a pre-trained **LightGBM model** to predict whether a credit card transaction is fraudulent.
Please enter the transaction details in the sidebar on the left to receive a prediction.
""")

st.sidebar.header("Transaction Input Features")

def get_user_input():
    """Creates sidebar widgets and collects user input."""
    st.sidebar.subheader("Transaction Details")
    category = st.sidebar.selectbox("Transaction Category", options=mappings['category_list'])
    amt = st.sidebar.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=10.0)
    trans_date = st.sidebar.date_input("Transaction Date", value=datetime.now())
    trans_time = st.sidebar.time_input("Transaction Time", value=datetime.now().time())

    st.sidebar.subheader("Cardholder Information")
    gender = st.sidebar.selectbox("Gender", options=mappings['gender_list'])
    age = st.sidebar.slider("Age", min_value=18, max_value=120, value=45)
    job = st.sidebar.selectbox("Job", options=mappings['job_list'])

    st.sidebar.subheader("Location Details")
    city = st.sidebar.selectbox("City", options=mappings['city_list'], index=100) # Default to a value
    state = st.sidebar.selectbox("State", options=mappings['state_list'], index=5) # Default to a value
    city_pop = st.sidebar.number_input("City Population", min_value=1, value=50000)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat = st.number_input("Cardholder Latitude", value=34.05, format="%.4f")
        long = st.number_input("Cardholder Longitude", value=-118.24, format="%.4f")
    with col2:
        merch_lat = st.number_input("Merchant Latitude", value=34.15, format="%.4f")
        merch_long = st.number_input("Merchant Longitude", value=-118.44, format="%.4f")

    # Combine date and time for processing
    trans_datetime = datetime.combine(trans_date, trans_time)

    # Store all inputs in a dictionary
    data = {
        'category': category, 'amt': amt, 'gender': gender, 'city': city,
        'state': state, 'lat': lat, 'long': long, 'city_pop': city_pop, 'job': job,
        'merch_lat': merch_lat, 'merch_long': merch_long, 'age': age,
        'trans_datetime': trans_datetime,
    }
    return data

input_data = get_user_input()

# --- 4. PREDICTION LOGIC AND DISPLAY ---

if st.sidebar.button("Detect Fraud", type="primary"):

    # 4.1. Feature Engineering on input data
    distance = haversine(input_data['lat'], input_data['long'], input_data['merch_lat'], input_data['merch_long'])
    trans_dt = input_data['trans_datetime']
    hour, day, month, weekday = trans_dt.hour, trans_dt.day, trans_dt.month, trans_dt.weekday()
    unix_time = int(trans_dt.timestamp())

    # 4.2. Apply saved mappings to categorical features
    category_encoded = mappings['category'][input_data['category']]
    gender_encoded = mappings['gender'][input_data['gender']]
    city_encoded = mappings['city'][input_data['city']]
    state_encoded = mappings['state'][input_data['state']]
    job_encoded = mappings['job'][input_data['job']]

    # 4.3. Create the final feature vector (DataFrame) for the model
    # The order of columns MUST exactly match the order used during training
    features = {
        'category': category_encoded, 'amt': input_data['amt'], 'gender': gender_encoded,
        'city': city_encoded, 'state': state_encoded, 'lat': input_data['lat'],
        'long': input_data['long'], 'city_pop': input_data['city_pop'], 'job': job_encoded,
        'unix_time': unix_time, 'merch_lat': input_data['merch_lat'],
        'merch_long': input_data['merch_long'], 'age': input_data['age'],
        'distance': distance, 'hour': hour, 'day': day, 'month': month, 'weekday': weekday
    }
    input_df = pd.DataFrame([features])
    
    # 4.4. Make prediction
    prediction_proba = model.predict(input_df, num_iteration=model.best_iteration)[0]

    # --- 5. DISPLAY RESULTS ---
    st.subheader("Prediction Result")
    
    # Apply the 0.5 threshold to classify
    is_fraud = prediction_proba > 0.5

    # Display prediction with appropriate styling and icons
    if is_fraud:
        st.error("🚨 FRAUD DETECTED 🚨", icon="🚨")
    else:
        st.success("✅ Transaction appears Legitimate ✅", icon="✅")

    # Display the probability score in a metric card
    st.metric(label="Fraud Probability Score", value=f"{prediction_proba:.4f}")
    st.progress(prediction_proba)

    st.info("""
    **Disclaimer:** This prediction is based on a machine learning model and should be used as a supplementary tool.
    A score greater than **0.5** is classified as potential fraud.
    """, icon="ℹ️")

    # Show the data that was used for the prediction
    with st.expander("View Input Data and Engineered Features"):
        st.json({k: str(v) for k, v in input_data.items()}) # Original inputs
        st.write("---")
        st.write("Features Sent to Model for Prediction:")
        st.dataframe(input_df) # Processed features

else:
    st.info("Please fill in the details on the left and click 'Detect Fraud' to get a prediction.")
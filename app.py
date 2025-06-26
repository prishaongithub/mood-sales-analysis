import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Page setup
st.set_page_config(page_title="MOODSpend Insights", layout="wide")
st.title("🎯 Customer Behavior Predictor")
st.markdown("This dashboard predicts how and when a customer is likely to spend based on mood, weather, season, and city.")

# Load models
spend_model = joblib.load("spend_amount_model.pkl")
category_model = joblib.load("category_model.pkl")
birthday_model = joblib.load("birthday_model.pkl")

# Load label encoder for Category model
category_encoder = joblib.load("label_encoder_category.pkl")

# Upload section
st.sidebar.header("📂 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📈 Data Preview")
    st.dataframe(df_uploaded.head())
    st.markdown("---")

    st.subheader("📊 Explore Data")
    num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

    if num_cols:
        col = st.selectbox("Numerical Column for Histogram", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df_uploaded[col], kde=True, ax=ax)
        st.pyplot(fig)

    if cat_cols:
        cat = st.selectbox("Categorical Column for Count Plot", cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(data=df_uploaded, x=cat, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.markdown("---")

# Helper: Check if near birthday
def is_near_birthday(transaction_date, birthdate):
    bday_this_year = birthdate.replace(year=transaction_date.year)
    delta = (bday_this_year - transaction_date).days
    return 0 <= delta <= 15

# Inputs
st.subheader("🧍 Customer Details & Prediction")

with st.form("form"):
    age = st.slider("Customer Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"])
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"])
    city = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'])
    category_input = st.selectbox("Product Category", ["Food", "Electronics", "Entertainment", "Personal Care", "Groceries"])
    transaction_date = st.date_input("Transaction Date", datetime.today())
    birthdate = st.date_input("Customer Birthdate", datetime(2000, 1, 1))
    submit = st.form_submit_button("🎯 Predict")

# Predictions
if submit:
    near_bday = 1 if is_near_birthday(transaction_date, birthdate) else 0

    # Model 1: Spend Prediction
    df_spend = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Season": [season],
        "Weather": [weather],
        "Is_Near_Birthday": [near_bday]
    })
    df_spend = pd.get_dummies(df_spend)
    df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
    predicted_amount = spend_model.predict(df_spend)[0]

    # Model 2: Category Prediction
    df_cat = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Weather": [weather],
        "Is_Near_Birthday": [near_bday],
        "Transaction Amount": [predicted_amount],
        "City": [city]
        })
    df_cat = pd.get_dummies(df_cat)
    df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
    predicted_category_code = category_model.predict(df_cat)[0]
    predicted_category = category_encoder.inverse_transform([predicted_category_code])[0]

    # --- Model 3: Predict Birthday Probability ---
    df_bday = pd.DataFrame({
    "Customer_Age": [age],
    "Gender": [gender],
    "Mood": [mood],
    "Season": [season],
    "Weather": [weather],
    "Category": [category_input]
    })
    df_bday = pd.get_dummies(df_bday)
    df_bday = df_bday.reindex(columns=birthday_model.feature_names_in_, fill_value=0)
    birthday_prob = birthday_model.predict_proba(df_bday)[0][1]

# 🔥 --- Boost Probability Based on Behavior Patterns ---
    boost = 0
    if gender == "Female" and mood == "Happy" and near_bday == 1:
        boost += 0.25
    if mood in ["Relaxed", "Happy"] and transaction_date.weekday() >= 5 and near_bday == 1:
        boost += 0.15
    if weather in ["Sunny", "Clear"] and near_bday == 1:
        boost += 0.10
    if category_input in ["Electronics", "Personal Care"] and near_bday == 1:
        boost += 0.10


    birthday_prob = min(birthday_prob + boost, 1.0)


    # Output
    st.subheader("📌 Results")
    st.metric("💸 Estimated Spend", f"₹{round(predicted_amount, 2)}")
    st.metric("📦 Likely Category", predicted_category)
    st.metric("🎂 Birthday Spend Probability", f"{round(birthday_prob * 100, 2)}%")

st.markdown("---")
st.caption("🚀 Built with ❤️ by Prisha Arora | Powered by Streamlit")

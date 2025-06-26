import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- Setup ---
st.set_page_config(page_title="MOODSpend Insights", layout="wide")
st.title("🧠 MOODSpend: Customer Behavior Analysis")
st.caption("🔍 Upload data, explore insights, and predict customer behavior in real-time.")

# --- Load Models ---
spend_model = joblib.load("spend_amount_model.pkl")
category_model = joblib.load("category_model.pkl")
birthday_model = joblib.load("birthday_model.pkl")
category_encoder = joblib.load("label_encoder_category.pkl")

# --- Helper: Near Birthday ---
def is_near_birthday(transaction_date, birthdate):
    bday_this_year = birthdate.replace(year=transaction_date.year)
    delta = (bday_this_year - transaction_date).days
    return 0 <= delta <= 15

# --- Section 1: Upload + EDA ---
st.header("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV to explore", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📈 Data Preview")
    st.dataframe(df_uploaded.head())

    with st.expander("📊 Explore Dataset"):
        num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

        if num_cols:
            col = st.selectbox("📉 Choose a Numerical Column", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df_uploaded[col], kde=True, ax=ax)
            st.pyplot(fig)

        if cat_cols:
            cat = st.selectbox("📊 Choose a Categorical Column", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df_uploaded, x=cat, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.markdown("---")

# --- Shared Inputs ---
st.header("🧍 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"])

with col2:
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"])
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    city = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'])

# --- Spend Amount Predictor ---
with st.expander("💸 Spend Amount Predictor", expanded=True):
    if st.button("Predict Spend Amount"):
        df_spend = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Season": [season],
            "Weather": [weather],
            "Is_Near_Birthday": [0]  # default
        })
        df_spend = pd.get_dummies(df_spend)
        df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
        predicted_amount = spend_model.predict(df_spend)[0]
        st.metric("Predicted Spend", f"₹{round(predicted_amount, 2)}")

# --- Category Predictor ---
with st.expander("📦 Category Predictor", expanded=True):
    transaction_amount = st.number_input("Transaction Amount", min_value=1.0, value=500.0)
    if st.button("Predict Category"):
        df_cat = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Weather": [weather],
            "Is_Near_Birthday": [0],  # assumed neutral
            "Transaction Amount": [transaction_amount],
            "City": [city]
        })
        df_cat = pd.get_dummies(df_cat)
        df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
        category_code = category_model.predict(df_cat)[0]
        category = category_encoder.inverse_transform([category_code])[0]
        st.success(f"They are most likely to buy from the category: **{category}**")

# --- Birthday Predictor ---
with st.expander("🎂 Birthday Spend Probability", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        transaction_date = st.date_input("Transaction Date", datetime.today())
    with col4:
        birthdate = st.date_input("Customer's Birthdate", datetime(2000, 1, 1))

    if st.button("Predict Birthday Purchase Probability"):
        near_bday = is_near_birthday(transaction_date, birthdate)

        df_bday = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Season": [season],
            "Weather": [weather],
            "Category": ["Electronics"]  # default input to maintain model input shape
        })

        df_bday = pd.get_dummies(df_bday)
        df_bday = df_bday.reindex(columns=birthday_model.feature_names_in_, fill_value=0)
        prob = birthday_model.predict_proba(df_bday)[0][1]

        # --- Boost Logic ---
        boost = 0
        if gender == "Female" and mood == "Happy" and near_bday:
            boost += 0.25
        if mood in ["Relaxed", "Happy"] and transaction_date.weekday() >= 5 and near_bday:
            boost += 0.20
        if weather in ["Sunny", "Clear"] and near_bday:
            boost += 0.15
        boost = min(boost, 0.45)

        final_prob = min(prob + boost, 1.0)

        st.metric("🎂 Birthday Spend Probability", f"{round(final_prob * 100, 2)}%")

# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ by Prisha Arora | Streamlit Deployment")

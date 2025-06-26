import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# 🔧 Page Setup
st.set_page_config(page_title="Customer Spend Intelligence", layout="wide")
st.title("🛍️ Customer Spending Intelligence Dashboard")
st.write("Upload your dataset or enter customer details to generate insights and predictions.")

# 🔹 Load Models
spend_model = joblib.load("spend_amount_model.pkl")      # RandomForestRegressor
category_model = joblib.load("category_model.pkl")       # XGBoost Classifier
birthday_model = joblib.load("birthday_model.pkl")       # Logistic Regression

# 🔸 Upload Section + EDA
st.subheader("📂 Upload Your Own Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file to explore and visualize your data", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("### 🧾 Data Preview", df_uploaded.head())
    st.write("Rows:", df_uploaded.shape[0], " | Columns:", df_uploaded.shape[1])
    st.write("### 📊 Statistical Summary")
    st.write(df_uploaded.describe())

    # Auto Visualizations
    num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

    if num_cols:
        selected_num = st.selectbox("Numerical column for distribution plot", num_cols)
        fig1, ax1 = plt.subplots()
        sns.histplot(df_uploaded[selected_num], kde=True, ax=ax1)
        st.pyplot(fig1)

    if cat_cols:
        selected_cat = st.selectbox("Categorical column for count plot", cat_cols)
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df_uploaded, x=selected_cat, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")

# 🎯 Function to determine if date is near birthday
def is_near_birthday(transaction_date, birthdate):
    bday_this_year = birthdate.replace(year=transaction_date.year)
    delta = (bday_this_year - transaction_date).days
    return 0 <= delta <= 15

# 🤖 Prediction Section
st.subheader("🧍 Predict Customer Spending Behavior")

with st.form("input_form"):
    st.markdown("### 📥 Enter Customer Information")

    age = st.slider("Customer Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"])
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"])
    city = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'])

    transaction_date = st.date_input("Transaction Date", value=datetime.today())
    birthdate = st.date_input("Customer's Birthdate", value=datetime(2000, 1, 1))

    category_input = st.selectbox("Category (for birthday model)", ["Apparel", "Food", "Electronics", "Entertainment"])

    submit = st.form_submit_button("Predict")

if submit:
    # Derived Feature
    birthday_flag = 1 if is_near_birthday(transaction_date, birthdate) else 0

    # --- Model 1: Predict Spend Amount ---
    df_spend = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Season": [season],
        "Weather": [weather],
        "Is_Near_Birthday": [birthday_flag]
    })
    df_spend = pd.get_dummies(df_spend)
    df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
    predicted_amount = spend_model.predict(df_spend)[0]

    # --- Model 2: Predict Category ---
    df_cat = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Weather": [weather],
        "Is_Near_Birthday": [birthday_flag],
        "Transaction Amount": [predicted_amount],
        "City": [city]
    })
    df_cat = pd.get_dummies(df_cat)
    df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
    predicted_category = category_model.predict(df_cat)[0]

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

    # --- Display Results ---
    st.subheader("🧠 Prediction Results")
    st.success(f"💸 Estimated Spend Amount: ₹{round(predicted_amount, 2)}")
    st.info(f"📦 They are most likely to buy from the category: **{predicted_category}**")
    st.warning(f"🎂 Probability of Spending Near Birthday: {round(birthday_prob * 100, 2)}%")

st.markdown("---")
st.markdown("<center><sub>✨ Made with ❤️ by Prisha Arora</sub></center>", unsafe_allow_html=True)

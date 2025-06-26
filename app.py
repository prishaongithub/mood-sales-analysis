import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 🔧 Page Configuration
st.set_page_config(page_title="🛒 MoodSpend AI Analysis Dashboard", layout="wide")
st.title("📈 Customer Behavior Intelligence")
st.markdown("Get insights, predictions, and visualizations based on customer profiles and behavior using machine learning models.")

# 🔹 Load Trained Models
spend_model = joblib.load("spend_amount_model.pkl")      # RandomForestRegressor
category_model = joblib.load("category_model.pkl")       # XGBoost Classifier
birthday_model = joblib.load("birthday_model.pkl")       # Logistic Regression

# 🔸 Upload Section
st.subheader("📂 Upload Your Customer Dataset")
uploaded_file = st.file_uploader("Upload a CSV file to analyze customer data", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df_uploaded.head())

    st.markdown("**Rows:** {} | **Columns:** {}".format(df_uploaded.shape[0], df_uploaded.shape[1]))

    # Summary
    st.markdown("### 📊 Statistical Summary")
    st.write(df_uploaded.describe())

    # Auto Visuals
    num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

    if num_cols:
        selected_num = st.selectbox("📉 Select a numerical column to visualize", num_cols)
        fig1, ax1 = plt.subplots()
        sns.histplot(df_uploaded[selected_num], kde=True, ax=ax1)
        st.pyplot(fig1)

    if cat_cols:
        selected_cat = st.selectbox("📊 Select a categorical column to visualize", cat_cols)
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df_uploaded, x=selected_cat, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")

# 🔮 Prediction Section
st.subheader("🤖 Predict Customer Behavior")

with st.form("prediction_form"):
    st.markdown("### 🧍 Enter Customer Information")
    
    age = st.slider("Customer Age", 15, 75, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy"])
    season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Spring"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Foggy"])
    category_input = st.selectbox("Likely Purchase Category", ["Clothing", "Cosmetic", "Electronics", "Market", "Restaurant", "Travel"])
    city = st.selectbox("City", ["Tokyo", "London", "New York", "Paris", "Berlin", "Toronto", "Sydney", "Dubai"])
    is_near_birthday = st.selectbox("Is it near their birthday?", ["Yes", "No"])
    
    submit = st.form_submit_button("🔍 Predict")

if submit:
    is_birthday = 1 if is_near_birthday == "Yes" else 0

    # Model 1: Predict Spend Amount
    df_spend = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Season": [season],
        "Weather": [weather],
        "Is_Near_Birthday": [is_birthday]
    })
    df_spend = pd.get_dummies(df_spend)
    df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
    predicted_amount = spend_model.predict(df_spend)[0]

    # Model 2: Predict Purchase Category
    df_cat = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Weather": [weather],
        "Is_Near_Birthday": [is_birthday],
        "Transaction Amount": [predicted_amount],
        "City": [city]
    })
    df_cat = pd.get_dummies(df_cat)
    df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
    predicted_category = category_model.predict(df_cat)[0]

    # Model 3: Predict Birthday Probability
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

    # Display Results
    st.markdown("## 🧠 Prediction Results")
    st.success(f"💸 Estimated Spend Amount: ₹{round(predicted_amount, 2)}")
    st.info(f"📦 Predicted Purchase Category: **{predicted_category}**")
    st.warning(f"🎂 Probability of Birthday Spending: **{round(birthday_prob * 100, 2)}%**")

st.markdown("---")
st.markdown("<center><sub>✨ Made with ❤️ by Prisha Arora</sub></center>", unsafe_allow_html=True)

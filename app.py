import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Influencers", layout="wide")

st.title("📊 Mood, Gender & Weather Influence on Sales")
st.write("Explore how mood, gender, and weather affect sales across categories.")

uploaded = st.file_uploader("Upload your cleaned dataset (.xlsx or .csv)", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("Raw Data")
    st.write(df.head())

    try:
        st.subheader("Mood vs Sales")
        mood_sales = df.groupby("Mood")["Sales"].sum().reset_index()
        fig1 = sns.barplot(data=mood_sales, x="Mood", y="Sales")
        st.pyplot(fig1.figure)

        st.subheader("Gender vs Sales")
        gender_sales = df.groupby("Gender")["Sales"].sum().reset_index()
        fig2 = sns.barplot(data=gender_sales, x="Gender", y="Sales")
        st.pyplot(fig2.figure)

        st.subheader("Weather vs Sales")
        weather_sales = df.groupby("Weather")["Sales"].sum().reset_index()
        fig3 = sns.barplot(data=weather_sales, x="Weather", y="Sales")
        st.pyplot(fig3.figure)

        st.subheader("Category-wise Breakdown by Mood")
        mood_cat = df.groupby(["Mood", "Category"])["Sales"].sum().reset_index()
        fig4 = sns.catplot(data=mood_cat, x="Mood", y="Sales", hue="Category", kind="bar", height=5, aspect=2)
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Something went wrong while plotting. Error: {e}")
else:
    st.info("Please upload your dataset to begin.")

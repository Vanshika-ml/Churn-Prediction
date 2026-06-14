import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

model = joblib.load(
    "models/churn_model.pkl"
)

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)
st.sidebar.title("Dashboard Info")

st.sidebar.metric(
    "Model Accuracy",
    "79%"
)

st.sidebar.info(
    """
    Customer Churn Prediction System

    Model: Random Forest

    Features:
    - Batch Prediction
    - Risk Analysis
    - PDF Reports
    """
)

st.title(
    "📊 Customer Churn Prediction Dashboard"
)

st.caption(
    "Machine Learning Powered Customer Retention Analytics"
)



st.divider()

st.header("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    df.dropna(inplace=True)
    encoders = joblib.load("models/encoders.pkl")
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    st.subheader("Uploaded Data")

    st.dataframe(df.head())
    if "customerID" in df.columns:
        df = df.drop("customerID",axis=1)
    if "Churn" in df.columns:
        df = df.drop("Churn",axis=1)    

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)[:, 1]

    df["Churn Prediction"] = predictions

    df["Probability"] = (
        probabilities * 100
    ).round(2)

    df["Risk"] = df["Probability"].apply(
        lambda x:
        "High"
        if x > 70
        else (
            "Medium"
            if x > 40
            else "Low"
        )
    )

    col1, col2 = st.columns(2)

    col1.metric(
       "High Risk %",
        f"{(len(df[df['Risk']=='High'])/len(df))*100:.2f}%"
    )

    col2.metric(
       "Low Risk %",
       f"{(len(df[df['Risk']=='Low'])/len(df))*100:.2f}%"
    )

    

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(df)

    high_risk = df[
        df["Risk"] == "High"
    ]

    st.subheader(
        "Top High Risk Customers"
    )

    st.dataframe(high_risk)

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "📥Download Prediction Results",
        csv,
        "predictions.csv",
        "text/csv"
    )
    st.subheader(
        "Risk Distribution"
    )

    risk_counts = (
        df["Risk"]
        .value_counts()
    )

    fig, ax = plt.subplots()

    ax.pie(
       risk_counts,
       labels=risk_counts.index,
       autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.subheader(
    "Feature Importance"
    )

    importance = model.feature_importances_

    feature_df = pd.DataFrame(
       {
         "Feature": df.columns[:len(importance)],
         "Importance": importance
        }
    )

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )
 
    fig, ax = plt.subplots()

    ax.barh( 
       feature_df["Feature"],
       feature_df["Importance"]
    )

    st.pyplot(fig)
    st.subheader(
    "Top 10 High Risk Customers"
    )

    top10 = df.sort_values(
      by="Probability",
      ascending=False
    ).head(10)

    st.dataframe(top10)

    pdf_file = "churn_report.pdf"

    c = canvas.Canvas(pdf_file)

    c.drawString(
       100,
       800,
      "Customer Churn Report"
    )

    c.drawString(
       100,
       770,
       f"Total Customers: {len(df)}"
    )

    c.drawString(
       100,
       740,
       f"High Risk Customers: {len(df[df['Risk']=='High'])}"
    )

    c.drawString(
       100,
       710,
       f"Average Probability: {df['Probability'].mean():.2f}%"
    )

    c.save()

    with open(pdf_file, "rb") as f:

        st.download_button(
          "Download PDF Report",
           f,
           file_name="Churn_Report.pdf"
        )
        st.caption(
            "Built with Python, Streamlit, Scikit-Learn and Pandas"
        )
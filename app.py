# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Distress Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

DATA_PATH = "Data/Financial Distress.csv"

df_app = pd.read_csv(DATA_PATH)

df_app["Distress_Class"] = np.where(
    df_app["Financial Distress"] <= -0.50,
    1,
    0
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

model = joblib.load("model.pkl")


# ============================================================
# 5. MAIN TITLE
# ============================================================

st.title("Financial Distress Prediction")
st.caption("Machine Learning–Based Early Warning System")


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Financial Risk Dashboard")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to",
    ["Business Problem", "Data Insights", "Prediction"]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning for Business Applications"
)


# ============================================================
# 7. BUSINESS PROBLEM PAGE
# ============================================================

if page == "Business Problem":

    st.header("📌 Business Problem")

    st.markdown(
        "### Early Identification of Financial Distress"
    )
    st.subheader("Business Problem")

    st.write(
        "Financial institutions and investors need an early-warning "
        "mechanism to identify companies that may be experiencing "
        "financial distress. Early identification can support closer "
        "monitoring and further financial assessment."
    )

    st.subheader("Analytics Objective")

    st.write(
        "The objective is to classify companies into healthy and "
        "financially distressed categories using available financial "
        "and non-financial characteristics."
    )

    st.subheader("Target Variable")

    st.write(
        "Financial Distress is converted into a binary classification "
        "target. A value less than or equal to -0.50 is classified as "
        "financially distressed (1), while a value greater than -0.50 "
        "is classified as healthy (0)."
    )

    st.subheader("Business Decision")

    st.write(
        "The prediction can be used as an early-warning indicator "
        "for additional financial analysis, closer monitoring, "
        "credit review, and investment due diligence."
    )

    st.subheader("Dataset")

    st.write("Financial Distress Prediction dataset")
    st.write("Observations: 3,672")
    st.write("Companies: 422")
    st.write("Original variables: 86")
    st.write("Time periods: 1–14")


# ============================================================
# 8. DATA INSIGHTS PAGE
# ============================================================

elif page == "Data Insights":

    st.header("📈 Data Insights")

    st.markdown(
        "Explore the dataset structure, financial distress distribution "
        "and model-related feature patterns."
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Observations",
            f"{len(df_app):,}"
        )

    with col2:
        st.metric(
            "Companies",
            df_app["Company"].nunique()
        )

    with col3:
        st.metric(
            "Distressed Cases",
            int(df_app["Distress_Class"].sum())
        )


    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Dataset Overview")

    overview = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Unique Companies",
            "Time Periods",
            "Missing Values",
            "Duplicate Rows"
        ],
        "Value": [
            df_app.shape[0],
            df_app.shape[1],
            df_app["Company"].nunique(),
            df_app["Time"].nunique(),
            int(df_app.isnull().sum().sum()),
            int(df_app.duplicated().sum())
        ]
    })

    st.dataframe(
        overview,
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Financial Distress Class Distribution")

    class_counts = (
        df_app["Distress_Class"]
        .value_counts()
        .sort_index()
    )

    class_labels = ["Healthy", "Financially Distressed"]

    fig, ax = plt.subplots()

    ax.bar(
        class_labels,
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Observations")
    ax.set_xlabel("Financial Status")
    ax.set_title("Distribution of Financial Distress Classes")

    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "The dataset is highly imbalanced, with healthy observations "
        "substantially outnumbering financially distressed observations. "
        "Therefore, accuracy alone is not sufficient for evaluating the "
        "classification model."
    )

    # --------------------------------------------------------
    # TIME-WISE DISTRESS RATE
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Financial Distress Rate Across Time")

    time_distress = (
        df_app.groupby("Time")["Distress_Class"]
        .mean()
        .reset_index()
    )

    time_distress["Distress_Rate"] = (
        time_distress["Distress_Class"] * 100
    )

    fig, ax = plt.subplots()

    ax.plot(
        time_distress["Time"],
        time_distress["Distress_Rate"],
        marker="o"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Distress Rate (%)")
    ax.set_title("Financial Distress Rate Across Time")

    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The chart shows how the observed distress rate varies across "
        "the available time periods. These patterns should be interpreted "
        "as descriptive relationships rather than causal effects."
    )

    # --------------------------------------------------------
    # FEATURE COEFFICIENT CHART
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Selected Model Feature Coefficients")

    coefficients = pd.Series({
        "x10": -2.517405,
        "x44": -1.677187,
        "x36": -1.001839,
        "x7": -0.623541,
        "x33": -0.225216
    }).sort_values()

    fig, ax = plt.subplots()

    coefficients.plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Coefficients in the Reduced Model")

    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Coefficient magnitude indicates the relative contribution of "
        "each selected standardized feature within the logistic regression "
        "model. A coefficient represents association in the model and "
        "should not be interpreted as proof of causation."
    )

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Key Business Insights")

    st.markdown(
        """
        - The dataset contains a large majority of healthy observations,
          creating a substantial class imbalance.
        - The distress rate varies across the available time periods.
        - The reduced Logistic Regression model uses five selected
          financial indicators: x36, x33, x44, x7 and x10.
        - Among these selected variables, x10 has the largest absolute
          model coefficient.
        - Because the objective is early identification of distressed
          companies, recall is an important evaluation metric.
        """
    )


# ============================================================
# 9. PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.header("🔍 Financial Distress Prediction")

    st.markdown(
        "Enter the selected financial indicator values to generate "
        "a model-based early-warning prediction."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.subheader("Enter Financial Indicators")

    x36 = st.number_input(
        "x36",
        min_value=float(df_app["x36"].min()),
        max_value=float(df_app["x36"].max()),
        value=float(df_app["x36"].median())
    )

    x33 = st.number_input(
        "x33",
        min_value=float(df_app["x33"].min()),
        max_value=float(df_app["x33"].max()),
        value=float(df_app["x33"].median())
    )

    x44 = st.number_input(
        "x44",
        min_value=float(df_app["x44"].min()),
        max_value=float(df_app["x44"].max()),
        value=float(df_app["x44"].median())
    )

    x7 = st.number_input(
        "x7",
        min_value=float(df_app["x7"].min()),
        max_value=float(df_app["x7"].max()),
        value=float(df_app["x7"].median())
    )

    x10 = st.number_input(
        "x10",
        min_value=float(df_app["x10"].min()),
        max_value=float(df_app["x10"].max()),
        value=float(df_app["x10"].median())
    )

    # --------------------------------------------------------
    # INPUT PREVIEW
    # --------------------------------------------------------

    st.subheader("Selected Input Values")

    input_preview = pd.DataFrame({
        "Feature": ["x36", "x33", "x44", "x7", "x10"],
        "Value": [x36, x33, x44, x7, x10]
    })

    st.dataframe(
        input_preview,
        hide_index=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Financial Distress",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "x36": [x36],
            "x33": [x33],
            "x44": [x44],
            "x7": [x7],
            "x10": [x10]
        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        # ----------------------------------------------------
        # PREDICTION METRICS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                "Financially Distressed" if prediction == 1 else "Healthy"
            )

        with col2:
            st.metric(
                "Distress Probability",
                f"{probability:.2%}"
            )

        # ----------------------------------------------------
        # PROBABILITY VISUALIZATION
        # ----------------------------------------------------

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Class": [
                "Healthy",
                "Financially Distressed"
            ],
            "Probability": [
                1 - probability,
                probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Class")
        )

        # ----------------------------------------------------
        # INTERPRETATION / RECOMMENDED ACTION
        # ----------------------------------------------------

        if prediction == 1:

            st.error("Financially Distressed")

            st.write(
                f"Probability of Financial Distress: "
                f"{probability:.2%}"
            )

            st.warning(
                "Recommended Action: Conduct additional financial "
                "analysis, monitor the company closely, and consider "
                "further credit or investment review."
            )

        else:

            st.success("Healthy")

            st.write(
                f"Probability of Financial Distress: "
                f"{probability:.2%}"
            )

            st.info(
                "Recommended Action: Continue normal monitoring, "
                "while considering other financial and business "
                "information."
            )
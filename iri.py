import streamlit as st
import pickle

# --- Page Config ---
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# --- Sidebar: Theme Customization ---
st.sidebar.header("🎨 Customize Theme")
bg_color = st.sidebar.color_picker("Background Color", "#ffffff")
title_color = st.sidebar.color_picker("Title Text Color", "#000000")
text_color = st.sidebar.color_picker("Text Color", "#000000")
button_color = st.sidebar.color_picker("Button Color", "#4CAF50")
button_text_color = st.sidebar.color_picker("Button Text Color", "#ffffff")
expander_color = st.sidebar.color_picker("Expander Background Color", "#f0f0f0")
expander_text_color = st.sidebar.color_picker("Expander Text Color", "#000000")

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background-color: {bg_color};
    }}

    /* App title */
    .stMarkdown h1 {{
        color: {title_color};
        text-align: center;
    }}

    /* Predict button */
    div.stButton > button {{
        background-color: {button_color};
        color: {button_text_color};
        border-radius: 5px;
        height: 40px;
        width: 150px;
        font-size:16px;
        font-weight:bold;
    }}
    div.stButton > button:hover {{
        opacity: 0.85;
    }}

    /* Expander header */
    .stExpander > div:first-child {{
        background-color: {expander_color} !important;
        color: {expander_text_color} !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 5px 10px;
    }}

    /* Expander content */
    .streamlit-expanderContent {{
        color: {text_color};
        padding: 10px;
    }}

    /* Input labels */
    label {{
        color: {text_color} !important;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load trained model ---
with open("model.pkl", "rb") as pickle_in:
    abd = pickle.load(pickle_in)

# --- App Title ---
st.markdown("<h1>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)

# --- Expander: How to use ---
with st.expander("ℹ️ How to use this app", expanded=True):
    st.markdown(
        "<p style='font-size:18px;'>"
        "Steps to use the app:<br>"
        "1. Measure the flower's Sepal and Petal length/width in cm.<br>"
        "2. Enter the values in the input boxes below.<br>"
        "3. Click the 'Predict' button to see the Iris species."
        "</p>",
        unsafe_allow_html=True
    )

# --- Input Fields ---
st.markdown("<h3>Enter Flower Measurements:</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# --- Prediction Mapping ---
label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# --- Predict Button ---
if st.button("Predict"):
    try:
        prediction = abd.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        species = label_map.get(prediction[0], "Unknown")
        st.markdown(
            f"<p style='color:{text_color}; font-size:18px;'>Predicted Species: <b>{species}</b></p>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error: {e}")

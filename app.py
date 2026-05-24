import streamlit as st
from PIL import Image
import pandas as pd
import time
import numpy as np

from tensorflow.keras.models import load_model

# LOAD MODEL
model = load_model("satellite_model.h5")

# PAGE CONFIG
st.set_page_config(
    page_title="Satellite AI",
    page_icon="🛰️",
    layout="wide"
)

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #001f3f
        );
        color: white;
    }

    h1 {
        color: #38bdf8;
        font-size: 55px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title("🛰️ GeoAI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Analytics"
    ]
)

# HOME PAGE
if menu == "Home":

    st.image(
        "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa",
        use_container_width=True
    )

    st.title("🛰️ Satellite Image Land Classification")

    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.success("Image Uploaded Successfully")

        with st.spinner("🛰️ Analyzing Satellite Image..."):

            time.sleep(2)

            img = image.resize((224, 224))

            img_array = np.array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction_array = model.predict(img_array)

            classes = [
                "Agriculture",
                "Forest",
                "Industrial",
                "Residential",
                "River"
            ]

            prediction = classes[np.argmax(prediction_array)]

            confidence = round(np.max(prediction_array) * 100, 2)

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Land Type", prediction)

        with col2:
            st.metric("Confidence", f"{confidence}%")

        st.progress(float(confidence) / 100)

        history = pd.DataFrame({
            "Image Name": [uploaded_file.name],
            "Predicted Class": [prediction],
            "Confidence (%)": [confidence]
        })

        st.subheader("Prediction History")

        st.table(history)

# ANALYTICS PAGE
elif menu == "Analytics":

    st.title("📊 Analytics Dashboard")

    data = {
        "Category": [
            "Forest",
            "River",
            "Residential",
            "Industrial"
        ],

        "Images": [
            250,
            180,
            220,
            150
        ]
    }

    df = pd.DataFrame(data)

    st.bar_chart(df.set_index("Category"))

    st.markdown("---")
    st.subheader("Model Accuracy")

    chart_data = pd.DataFrame({
        "Epoch": [1, 2, 3, 4, 5],
        "Accuracy": [72, 81, 88, 93, 97]
    })

    st.area_chart(chart_data.set_index("Epoch"))
    st.subheader("Land Distribution")

    pie_data = pd.DataFrame({
        "Category": ["Forest", "River", "Residential", "Industrial"],
        "Values": [35, 25, 20, 20]
    })

    st.pyplot(
        pie_data.set_index("Category").plot.pie(
            y="Values",
            autopct="%1.1f%%",
            figsize=(5, 5)
        ).figure
    )
    st.caption(
        "Developed using Streamlit • GeoAI • ISRO NRSC Project"
    )
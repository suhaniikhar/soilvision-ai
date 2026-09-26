app.py


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SoilVision AI",
    page_icon="🌱",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ======================================================
   MAIN BACKGROUND
   ====================================================== */

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5,
        #bbf7d0
    );
}


/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {
    background: #064e3b !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}


/* ======================================================
   MAIN TITLE
   ====================================================== */

.title-bar {
    background: linear-gradient(
        90deg,
        #059669,
        #10b981,
        #14b8a6
    );

    padding: 20px;

    border-radius: 14px;

    text-align: center;

    font-size: 30px;

    font-weight: 800;

    color: white;

    margin-bottom: 25px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.15);
}


/* ======================================================
   RESULT DASHBOARD
   ====================================================== */

.result-box {
    background: rgba(255,255,255,0.85);

    padding: 28px;

    border-radius: 18px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.12);

    margin-top: 25px;

    border: 1px solid rgba(255,255,255,0.7);
}


/* ======================================================
   METRIC CARDS
   ====================================================== */

.metric-box {
    background: linear-gradient(
        135deg,
        #10b981,
        #06b6d4
    );

    color: white;

    padding: 22px;

    border-radius: 15px;

    text-align: center;

    font-size: 20px;

    font-weight: 700;

    min-height: 85px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    box-shadow:
        0 8px 20px rgba(0,0,0,0.15);

    transition: 0.3s;
}

.metric-box:hover {
    transform: translateY(-5px);

    box-shadow:
        0 12px 28px rgba(0,0,0,0.2);
}


/* ======================================================
   INFORMATION CARDS
   ====================================================== */

.info-box {
    background: #f0fdf4;

    padding: 17px;

    border-radius: 12px;

    margin-bottom: 12px;

    color: #064e3b;

    font-weight: 500;

    border-left: 5px solid #10b981;
}


/* ======================================================
   SECTION HEADERS
   ====================================================== */

.section-title {
    font-size: 22px;

    font-weight: 750;

    color: #064e3b;

    margin-bottom: 15px;
}


/* ======================================================
   HEALTH SCORE
   ====================================================== */

.progress-container {
    width: 100%;

    height: 24px;

    background: #d1d5db;

    border-radius: 15px;

    overflow: hidden;

    margin-top: 10px;

    margin-bottom: 15px;
}

.progress-fill {
    height: 100%;

    background: linear-gradient(
        90deg,
        #059669,
        #10b981,
        #22c55e
    );

    border-radius: 15px;

    text-align: right;

    padding-right: 8px;

    color: white;

    font-size: 13px;

    font-weight: 700;

    line-height: 24px;
}


/* ======================================================
   FILE UPLOADER
   ====================================================== */

[data-testid="stFileUploader"] {
    background: white;

    border-radius: 15px;

    padding: 20px;

    border: 2px dashed #10b981;

    box-shadow:
        0 7px 20px rgba(0,0,0,0.08);
}


/* ======================================================
   BUTTON
   ====================================================== */

.stButton > button {
    background: linear-gradient(
        90deg,
        #059669,
        #10b981
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 10px !important;

    padding: 10px 25px !important;

    font-weight: 700 !important;

    transition: 0.3s !important;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 18px rgba(0,0,0,0.2);
}


/* ======================================================
   HEADINGS
   ====================================================== */

h1, h2, h3, h4 {
    color: #064e3b !important;
}


/* ======================================================
   GENERAL TEXT
   ====================================================== */

p {
    color: #374151;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model(
    "soil_model.h5",
    compile=False
)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

MODEL_ACCURACY = 83.60


# ==========================================================
# SOIL CLASSES
# ==========================================================

classes = [
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil"
]


# ==========================================================
# SOIL INFORMATION
# ==========================================================

soil_info = {

    "Alluvial Soil":
        "Found mainly near river basins and known for high fertility.",

    "Arid Soil":
        "Dry soil commonly found in desert and semi-desert regions.",

    "Black Soil":
        "Clay-rich soil with high water retention and commonly associated with cotton cultivation.",

    "Laterite Soil":
        "Develops in tropical regions with high rainfall and intense weathering.",

    "Mountain Soil":
        "Found in hilly and mountainous regions and supports plantation crops.",

    "Red Soil":
        "Contains iron compounds that give it its characteristic reddish colour.",

    "Yellow Soil":
        "Generally associated with humid conditions and is related to red soil."
}


# ==========================================================
# SOIL QUALITY
# ==========================================================

soil_quality = {

    "Alluvial Soil":
        "High fertility and suitable for several agricultural crops.",

    "Arid Soil":
        "Low fertility with limited moisture availability.",

    "Black Soil":
        "High fertility and strong water-retention capacity.",

    "Laterite Soil":
        "Moderate fertility and may require nutrient management.",

    "Mountain Soil":
        "Moderate fertility and suitable for plantation crops.",

    "Red Soil":
        "Medium fertility and commonly suitable for pulses and millets.",

    "Yellow Soil":
        "Moderate fertility with suitability for selected crops."
}


# ==========================================================
# CROP RECOMMENDATIONS
# ==========================================================

soil_crops = {

    "Alluvial Soil":
        "Rice, Wheat, Sugarcane",

    "Arid Soil":
        "Millets, Barley",

    "Black Soil":
        "Cotton, Soybean",

    "Laterite Soil":
        "Tea, Coffee, Cashew",

    "Mountain Soil":
        "Tea, Coffee, Spices",

    "Red Soil":
        "Groundnut, Pulses",

    "Yellow Soil":
        "Maize, Groundnut"
}


# ==========================================================
# SOIL CHARACTERISTICS
# ==========================================================

soil_characteristics = {

    "Alluvial Soil": {
        "Color": "Light Brown",
        "Texture": "Loamy",
        "Water Retention": "Moderate",
        "Fertility": "High",
        "Drainage": "Good"
    },

    "Arid Soil": {
        "Color": "Pale Brown",
        "Texture": "Sandy",
        "Water Retention": "Very Low",
        "Fertility": "Low",
        "Drainage": "Very Fast"
    },

    "Black Soil": {
        "Color": "Dark Black",
        "Texture": "Clayey",
        "Water Retention": "High",
        "Fertility": "High",
        "Drainage": "Moderate"
    },

    "Laterite Soil": {
        "Color": "Reddish Brown",
        "Texture": "Gravelly",
        "Water Retention": "Low",
        "Fertility": "Moderate",
        "Drainage": "Fast"
    },

    "Mountain Soil": {
        "Color": "Dark Brown",
        "Texture": "Loamy",
        "Water Retention": "Moderate",
        "Fertility": "Moderate",
        "Drainage": "Good"
    },

    "Red Soil": {
        "Color": "Red",
        "Texture": "Sandy Loam",
        "Water Retention": "Low",
        "Fertility": "Medium",
        "Drainage": "Fast"
    },

    "Yellow Soil": {
        "Color": "Yellowish",
        "Texture": "Sandy Loam",
        "Water Retention": "Low",
        "Fertility": "Medium",
        "Drainage": "Good"
    }
}


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("## 🌱 SoilVision AI")

st.sidebar.write(
    "AI-Based Soil Classification System"
)

st.sidebar.markdown("---")


# Developer information

st.sidebar.markdown("### 👨‍💻 Developed By")

st.sidebar.markdown("""
**Om S. Dhadse**  
**Suhani G. Kalsait**  
**Himani Raut**
""")


st.sidebar.markdown("---")


# Supervisor

st.sidebar.markdown("### 🎓 Project Supervisor")

st.sidebar.write(
    "Prof. V. V. Bais"
)


st.sidebar.markdown("---")


# Project information

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.write(
    "Deep Learning based soil image classification using CNN."
)


st.sidebar.markdown("---")


# Dataset

st.sidebar.markdown("### 📊 Dataset")

st.sidebar.write("Soil Classes: 7")
st.sidebar.write("Image Size: 128 × 128")
st.sidebar.write("Training: 80%")
st.sidebar.write("Validation: 20%")


st.sidebar.markdown("---")


# Model

st.sidebar.markdown("### 🧠 Model")

st.sidebar.write("CNN")
st.sidebar.write("TensorFlow / Keras")
st.sidebar.write("Accuracy: ~83.60%")


# ==========================================================
# MAIN TITLE
# ==========================================================

st.markdown(
    """
    <div class="title-bar">
        🌱 SoilVision - AI Soil Classification Dashboard
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Upload a soil image and let the AI model classify the soil type."
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "result" not in st.session_state:
    st.session_state.result = None


# ==========================================================
# IMAGE PROCESSING AND PREDICTION
# ==========================================================

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1.3, 1])

    # ---------------- IMAGE ----------------

    with col1:

        st.markdown(
            "### 🖼️ Uploaded Soil Image"
        )

        st.image(
            img,
            use_container_width=True
        )


    # ---------------- PREDICTION ----------------

    with col2:

        st.markdown(
            "### 🔍 Soil Analysis"
        )

        st.write(
            "Click the button below to analyze the uploaded image."
        )

        if st.button(
            "🌱 Predict Soil Type",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing soil using AI..."
            ):

                # Resize image
                img_resized = img.resize(
                    (128, 128)
                )

                # Convert to NumPy
                img_array = np.array(
                    img_resized
                )

                # Normalize
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Prediction
                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                # Select class
                result = classes[
                    np.argmax(prediction)
                ]

                # Save result
                st.session_state.result = result


# ==========================================================
# RESULT DASHBOARD
# ==========================================================

if st.session_state.result is not None:

    result = st.session_state.result

    char = soil_characteristics[result]


    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )


    st.markdown(
        "## 🌱 Soil Intelligence Dashboard"
    )


    # ======================================================
    # TOP METRICS
    # ======================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="metric-box">
                🌱<br>
                Predicted Soil<br>
                <strong>{result}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-box">
                🌿<br>
                Soil Fertility<br>
                <strong>{char["Fertility"]}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-box">
                🧠<br>
                Model Accuracy<br>
                <strong>{MODEL_ACCURACY}%</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ======================================================
    # MAIN INFORMATION
    # ======================================================

    left, right = st.columns(2)


    # ------------------------------------------------------
    # LEFT
    # ------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">📊 Soil Insights</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-box">
                <strong>Soil Information</strong><br><br>
                {soil_info[result]}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-box">
                <strong>Soil Quality</strong><br><br>
                {soil_quality[result]}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-box">
                <strong>🌾 Recommended Crops</strong><br><br>
                {soil_crops[result]}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ------------------------------------------------------
    # RIGHT
    # ------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">🌿 Soil Health</div>',
            unsafe_allow_html=True
        )


        score_map = {
            "High": 90,
            "Moderate": 65,
            "Medium": 60,
            "Low": 40
        }


        score = score_map.get(
            char["Fertility"],
            50
        )


        st.write(
            f"Estimated Soil Health: **{score}%**"
        )


        st.markdown(
            f"""
            <div class="progress-container">
                <div
                    class="progress-fill"
                    style="width:{score}%"
                >
                    {score}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-box">
                <strong>🤖 AI Insight</strong><br><br>

                The predicted soil is classified as
                <strong>{result}</strong>.

                It has
                <strong>{char["Fertility"]}</strong>
                fertility and may be suitable for:

                <strong>{soil_crops[result]}</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ======================================================
    # SOIL CHARACTERISTICS
    # ======================================================

    st.markdown(
        '<div class="section-title">🧾 Soil Characteristics</div>',
        unsafe_allow_html=True
    )


    characteristics = list(
        char.items()
    )


    c1, c2, c3 = st.columns(3)


    for i, (key, value) in enumerate(
        characteristics
    ):

        box = f"""
        <div class="info-box">
            <strong>{key}</strong><br>
            {value}
        </div>
        """


        if i % 3 == 0:

            c1.markdown(
                box,
                unsafe_allow_html=True
            )

        elif i % 3 == 1:

            c2.markdown(
                box,
                unsafe_allow_html=True
            )

        else:

            c3.markdown(
                box,
                unsafe_allow_html=True
            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "🌱 SoilVision AI | CNN-Based Soil Classification System"
)
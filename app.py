import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ===========================
# ===============================
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

/* Existing CSS stays here */
/* CAMERA BUTTON */
div.st-key-open_camera button {
    width: 75px !important;
    min-width: 75px !important;
    height: 65px !important;
    min-height: 65px !important;

    background: #059669 !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 0 !important;
}

/* ENLARGE CAMERA ICON INSIDE BUTTON */
div.st-key-open_camera button p {
    font-size: 48px !important;
    line-height: 1 !important;
    margin: 0 !important;
}

/* HOVER */
div.st-key-open_camera button:hover {
    background: #047857 !important;
}

/* Camera hover ONLY */
.st-key-open_camera button:hover {
    background-color: #047857 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

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
   PREDICT BUTTON
   ====================================================== */

.st-key-predict_soil button {
    width: 100% !important;
    min-height: 75px !important;
    height: 75px !important;

    font-size: 22px !important;
    font-weight: 800 !important;

    background: linear-gradient(
        90deg,
        #059669,
        #10b981
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 18px !important;

    padding: 15px 20px !important;

    transition: 0.3s !important;
}

.st-key-predict_soil button:hover {
    transform: translateY(-3px);

    box-shadow:
        0 8px 20px rgba(0,0,0,0.2);
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
    "AI-Based Soil Classification System and Crop Recommendation"
)

st.sidebar.markdown("---")


# Developer information

st.sidebar.markdown("### 👨‍💻 Developed By")

st.sidebar.markdown("""
**Suhani S. Ikhar**  
**Rahul S. Nishad**  
""")


st.sidebar.markdown("---")


# Supervisor

st.sidebar.markdown("### 🎓 Project Supervisor")

st.sidebar.write(
    "Prof. Samadhan Mandpe"
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
# COMPACT CAMERA ICON + GALLERY UPLOAD
# ==========================================================

st.subheader("📸 Soil Image Analysis")

st.write("Upload a soil image or capture one using the camera.")

# Remember whether the camera should be open
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# Upload bar and small camera button side by side
upload_col, camera_col = st.columns(
    [10, 2],
    vertical_alignment="bottom"
)

with upload_col:
    uploaded_image = st.file_uploader(
        "Choose a soil image",
        type=["jpg", "jpeg", "png"],
        key="soil_gallery"
    )

with camera_col:
    if st.button(
        "📷",
        type="secondary",
        help="Open camera",
        key="open_camera"
    ):
        st.session_state.show_camera = (
            not st.session_state.show_camera
        )
    

# Show the camera ONLY after clicking the icon
camera_image = None

if st.session_state.show_camera:

    st.markdown("#### Capture Soil Image")

    camera_image = st.camera_input(
        "Take a picture of the soil",
        key="soil_camera"
    )

# Select camera or gallery image
if camera_image is not None:
    image_file = camera_image

elif uploaded_image is not None:
    image_file = uploaded_image

else:
    image_file = None

# ==========================================================
# SESSION STATE
# ==========================================================

if "result" not in st.session_state:
    st.session_state.result = None

# ==========================================================
# IMAGE PROCESSING AND PREDICTION
# ==========================================================

if image_file is not None:
    try:
        # Open either the captured photo or uploaded image consistently.
        img = Image.open(image_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        # ---------------- IMAGE PREVIEW ----------------
        with col1:
            st.markdown("### 🖼️ Selected Soil Image")
            st.image(
                img,
                caption="Selected Soil Image",
                use_container_width=True
            )

        # ---------------- PREDICTION ----------------
        with col2:
            st.markdown("### 🔍 Soil Analysis")
            st.write("Click below to analyze the selected image.")

            if st.button(
                "🌱 Predict Soil Type",
                use_container_width=True,
                key="predict_soil"
            ):
                with st.spinner("Analyzing soil using AI..."):
                    # Resize to the input size used when training the CNN.
                    img_resized = img.resize((128, 128))

                    # Convert to float32, normalize, and add batch dimension.
                    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict soil class using the existing trained model.
                    prediction = model.predict(img_array, verbose=0)
                    result = classes[int(np.argmax(prediction, axis=1)[0])]

                    # Store result for the result dashboard below.
                    st.session_state.result = result

    except Exception as e:
        st.error(f"Could not read or analyze this image. Please try a JPG or PNG soil photo. Details: {e}")

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
        <strong>🤖Test AI Insight</strong><br><br>

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
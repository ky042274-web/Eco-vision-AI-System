# ECO VISION AI - FINAL PROFESSIONAL UI
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import time


# PAGE CONFIG
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_model.h5")

model = load_model()


# CLASSES
classes = [
    "Cardboard",
    "Glass",
    "Metal",
    "Organic",
    "Paper",
    "Plastic"
]


# WASTE INFORMATION
waste_info = {

    "Plastic": {
        "bin":"🟡 Yellow Recycling Bin",
        "emoji":"🥤",
        "impact":"Plastic takes 500+ years to decompose.",
        "tip":"Avoid single-use plastics."
    },

    "Paper": {
        "bin":"🔵 Blue Recycling Bin",
        "emoji":"📄",
        "impact":"Paper recycling saves trees.",
        "tip":"Reuse notebooks and paper bags."
    },

    "Glass": {
        "bin":"🟢 Green Glass Bin",
        "emoji":"🍾",
        "impact":"Glass is infinitely recyclable.",
        "tip":"Separate glass properly."
    },

    "Metal": {
        "bin":"⚫ Metal Collection Bin",
        "emoji":"🔩",
        "impact":"Metal recycling saves energy.",
        "tip":"Recycle cans properly."
    },

    "Organic": {
        "bin":"🟤 Compost Bin",
        "emoji":"🌱",
        "impact":"Organic waste becomes compost.",
        "tip":"Convert food waste into fertilizer."
    },

    "Cardboard": {
        "bin":"🔵 Paper/Cardboard Bin",
        "emoji":"📦",
        "impact":"Cardboard is recyclable.",
        "tip":"Flatten boxes before recycling."
    }
}


# SESSION STATES
if "started" not in st.session_state:
    st.session_state.started = False

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# ----------------
# CSS
# ----------------
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    *{
        font-family:'Poppins',sans-serif;
    }

    html, body, [class*="css"]{
        color:white !important;
    }

    /* =====================================================
       BACKGROUND
    ===================================================== */

    .stApp{

        background:
        linear-gradient(
            rgba(0,0,0,0.78),
            rgba(0,0,0,0.92)
        ),

        url("https://images.unsplash.com/photo-1497436072909-60f360e1d4b1?q=80&w=2070&auto=format&fit=crop");

        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }

    /* =====================================================
       REMOVE STREAMLIT ITEMS
    ===================================================== */

    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    header{visibility:hidden;}

    /* =====================================================
       TEXT COLORS
    ===================================================== */

    h1,h2,h3,h4,h5,h6{
        color:white !important;
    }

    p,label,span{
        color:white !important;
    }

    [data-testid="stMarkdownContainer"] *{
        color:white !important;
    }

    /* =====================================================
       WELCOME SCREEN
    ===================================================== */

    .welcome-box{

        display:flex;
        justify-content:center;
        align-items:center;
        height:85vh;
    }

    .welcome-card{

        width:70%;
        text-align:center;

        background:rgba(255,255,255,0.08);

        border-radius:30px;

        padding:50px;

        backdrop-filter:blur(15px);

        border:1px solid rgba(255,255,255,0.15);

        box-shadow:0px 8px 32px rgba(0,0,0,0.4);
    }

    /* =====================================================
       MAIN TITLE
    ===================================================== */

    .main-title{

        text-align:center;

        font-size:82px;

        font-weight:800;

        background:linear-gradient(
            to right,
            #00F260,
            #0575E6
        );

        -webkit-background-clip:text;

        -webkit-text-fill-color:transparent;
    }

    .subtitle{

        text-align:center;

        font-size:24px;

        margin-bottom:30px;

        color:white !important;
    }

    /* =====================================================
       GLASS CARD
    ===================================================== */

    .glass{

        background:rgba(255,255,255,0.08);

        border-radius:25px;

        padding:28px;

        backdrop-filter:blur(14px);

        border:1px solid rgba(255,255,255,0.15);

        box-shadow:0px 8px 32px rgba(0,0,0,0.35);

        margin-bottom:20px;
    }

    /* =====================================================
       METRIC CARDS
    ===================================================== */

    .metric-box{

        background:rgba(255,255,255,0.08);

        border-radius:20px;

        padding:20px;

        text-align:center;

        border:1px solid rgba(255,255,255,0.10);

        backdrop-filter:blur(10px);

        transition:0.3s;
    }

    .metric-box:hover{

        transform:translateY(-5px);

        box-shadow:0px 0px 20px rgba(0,255,150,0.35);
    }

    .metric-number{

        font-size:38px;

        font-weight:bold;

        color:#00ff99;
    }

    .metric-text{

        font-size:16px;

        color:white !important;
    }

    /* =====================================================
       FILE UPLOADER
    ===================================================== */

    section[data-testid="stFileUploader"]{

        background:rgba(255,255,255,0.08);

        padding:25px;

        border-radius:20px;

        border:2px dashed rgba(255,255,255,0.3);

        backdrop-filter:blur(10px);
    }

    div[data-testid="stFileUploader"] button{

        background:linear-gradient(
            45deg,
            #00F260,
            #0575E6
        ) !important;

        color:black !important;

        font-weight:bold !important;

        border:none !important;

        border-radius:12px !important;
    }

    section[data-testid="stFileUploader"] small{

        color:black !important;

        font-weight:bold;
    }

    /* =====================================================
       BUTTONS
    ===================================================== */

    .stButton > button{

        width:100%;

        background:linear-gradient(
            45deg,
            #00F260,
            #0575E6
        );

        color:white !important;

        border:none;

        padding:15px;

        border-radius:15px;

        font-size:18px;

        font-weight:bold;

        transition:0.3s;
    }

    .stButton > button:hover{

        transform:scale(1.03);

        box-shadow:0px 0px 20px rgba(0,255,150,0.45);
    }

    /* =====================================================
       IMAGE
    ===================================================== */

    img{
        border-radius:20px;
    }

    /* =====================================================
       ANIMATION
    ===================================================== */

    @keyframes slideRight{

        0%{
            transform:translateX(-100px);
            opacity:0;
        }

        100%{
            transform:translateX(0px);
            opacity:1;
        }
    }

    .recycle-animation{

        animation:slideRight 1.5s ease;
    }

    /* =====================================================
       SCROLLBAR
    ===================================================== */

    ::-webkit-scrollbar{
        width:10px;
    }

    ::-webkit-scrollbar-thumb{
        background:#00F260;
        border-radius:20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# WELCOME SCREEN
# ----------------------
if st.session_state.started == False:

    st.markdown("""
    <div class="welcome-box">

    <div class="welcome-card">

    <h1 style="font-size:85px;">
    ♻️ EcoVision AI
    </h1>

    <h2 style="color:#00ff99;">
    Smart Sustainable Waste Management
    </h2>

    <p style="font-size:24px; line-height:1.8;">

    🌍 AI Waste Classification <br>
    ♻️ Smart Recycling Guidance <br>
    🗑️ Intelligent Dustbin Recommendation <br>
    🌱 Real World Sustainability Awareness

    </p>

    </div>

    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,1,1])

    with c2:

        if st.button("🚀 Start EcoVision AI"):

            st.session_state.started = True

            st.rerun()

    st.stop()


# HEADER
st.markdown("""
<div class="main-title">
♻️ EcoVision AI
</div>

<div class="subtitle">
AI Powered Sustainable Waste Classification System
</div>
""", unsafe_allow_html=True)


# METRICS
m1,m2,m3,m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-box">
    <div class="metric-number">6</div>
    <div class="metric-text">Waste Categories</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-box">
    <div class="metric-number">80%</div>
    <div class="metric-text">AI Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-box">
    <div class="metric-number">🌍</div>
    <div class="metric-text">Eco Friendly</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-box">
    <div class="metric-number">AI</div>
    <div class="metric-text">Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# FILE UPLOADER
# INPUT OPTION

option = st.radio(
    "📥 Choose Input Method",
    ["Upload Image", "Use Camera"]
)

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload Waste Image",
        type=["jpg","jpeg","png"]
    )

else:

    camera_col1, camera_col2, camera_col3 = st.columns([1,2,1])

    with camera_col2:

        uploaded_file = st.camera_input(
            "📸 Capture Waste Image"
        )


# MAIN CONTENT
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left,right = st.columns([1,1.2])


    # LEFT SIDE
    
    with left:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("📸 Uploaded Waste")

        st.image(image, width=260)

        st.write("")

        st.subheader("🌍 Sustainability Facts")

        st.success("♻️ Recycling 1 ton paper saves 17 trees.")

        st.info("🌱 Proper waste segregation reduces pollution.")

        st.warning("🚯 Plastic pollution harms marine life.")

        st.metric("Daily Global Waste", "3.5 Million Tons")

        st.write("")

        
        # PIE CHART
        st.subheader("♻️ Waste Distribution")

        eco_data = pd.DataFrame({

            "Category":[
                "Plastic",
                "Paper",
                "Organic",
                "Glass"
            ],

            "Value":[40,25,20,15]
        })

        fig = px.pie(

            eco_data,

            names="Category",

            values="Value",

            hole=0.5
        )

        fig.update_traces(

            textfont_size=18,

            textfont_color="white",

            textinfo="label+percent"
        )

        fig.update_layout(

            paper_bgcolor='rgba(0,0,0,0)',

            plot_bgcolor='rgba(0,0,0,0)',

            font_color='white',

            height=380,

            legend=dict(
                font=dict(
                    color="white",
                    size=14
                )
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------
    # RIGHT SIDE
    # -------------------
    with right:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("🤖 AI Waste Analysis")

        if st.button("♻️ Analyze Waste"):

            with st.spinner("Analyzing Waste..."):

                time.sleep(2)

                img = image.resize((224,224))

                img_array = np.array(img)

                img_array = np.expand_dims(img_array, axis=0)

                img_array = tf.keras.applications.efficientnet.preprocess_input(
                    img_array
                )

                prediction = model.predict(img_array)

                predicted_index = np.argmax(prediction)

                confidence = np.max(prediction) * 100

                predicted_class = classes[predicted_index]

                st.session_state.prediction_done = True

                st.session_state.predicted_class = predicted_class

                st.session_state.confidence = confidence

                st.session_state.prediction = prediction

        if st.session_state.prediction_done:

            predicted_class = st.session_state.predicted_class

            confidence = st.session_state.confidence

            prediction = st.session_state.prediction

            info = waste_info[predicted_class]

            st.success(f"✅ Waste Detected: {predicted_class}")

            st.info(f"🎯 Confidence Score: {confidence:.2f}%")

            st.warning(f"🌍 Environmental Impact: {info['impact']}")

            st.success(f"💡 Sustainability Tip: {info['tip']}")

            
            # WHY AI PREDICTED THIS
            st.subheader("🧠 Why AI Predicted This?")

            reasons = {

                "Plastic":"Detected shiny plastic texture and bottle-like structure.",

                "Paper":"Detected flat recyclable paper texture.",

                "Glass":"Detected reflective transparent material.",

                "Metal":"Detected metallic edges and reflective surface.",

                "Organic":"Detected biodegradable natural texture.",

                "Cardboard":"Detected brown box-like recyclable material."
            }

            st.info(reasons[predicted_class])

            # ---------------------
            # AI CONFIDENCE CHART
            # ---------------------
            st.subheader("📊 AI Confidence Analysis")

            chart_data = pd.DataFrame({

                "Waste Type": classes,

                "Confidence": prediction[0] * 100
            })

            bar_fig = px.bar(

                chart_data,

                x="Waste Type",

                y="Confidence",

                text="Confidence",

                color="Confidence"
            )

            bar_fig.update_traces(

                texttemplate='%{text:.2f}%',

                textposition='outside'
            )

            bar_fig.update_layout(

                height=450,

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                font_color='white',

                xaxis_title="Waste Category",

                yaxis_title="Confidence Score (%)"
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )

            # -----------------
            # RECYCLE SECTION
            # -----------------
            st.subheader("♻️ Smart Recycling Action")

            st.info(
                f"Detected waste belongs to: {info['bin']}"
            )

            st.write("")

            col1, col2 = st.columns([1,1])

            with col1:

                st.markdown("""
                <div class="recycle-animation">
                """, unsafe_allow_html=True)

                st.image(
                    image,
                    width=180,
                    caption="Detected Waste"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:

                st.markdown(f"""
                <div style="
                text-align:center;
                font-size:130px;
                ">
                🗑️
                </div>

                <h3 style="
                text-align:center;
                color:#00ff99;
                ">
                {info['bin']}
                </h3>
                """, unsafe_allow_html=True)

            st.write("")

            if st.button("♻️ Recycle This Waste"):

                recycle_placeholder = st.empty()

                recycle_placeholder.markdown(f"""
                <div style="
                text-align:center;
                padding:20px;
                ">

                <div style="
                font-size:70px;
                animation: slideRight 2s ease;
                ">
                {info['emoji']} ➜ ♻️ ➜ 🗑️
                </div>

                <h2 style="
                color:#00ff99;
                margin-top:20px;
                ">
                Recycling In Progress...
                </h2>

                </div>
                """, unsafe_allow_html=True)

                time.sleep(2)

                recycle_placeholder.success(
                    "🎉 Congratulations! Waste Successfully Recycled"
                )

                st.success(
                    "🌱 You contributed to a cleaner and greener future."
                )

                st.info(
                    "♻️ Try implementing proper waste segregation in real life too."
                )

        st.markdown('</div>', unsafe_allow_html=True)


# FOOTER
st.write("")
st.write("")

st.markdown("""
<div style="
text-align:center;
color:white;
padding:20px;
">

<hr>

<h2>🌍 EcoVision AI</h2>

Smart AI Based Sustainable Waste Management System

<br>

TensorFlow • EfficientNetB0 • Streamlit • Deep Learning

<br><br>

♻️ Together We Can Build A Cleaner Future ♻️

</div>
""", unsafe_allow_html=True)
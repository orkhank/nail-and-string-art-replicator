import streamlit as st
import cv2 as cv
from main import DNA

st.title("Nail and String Art")
image = cv.imread("image.jpg")
with st.sidebar:
    st.write("Parameters")
    st.slider(
        "Radius",
        0,
        image.shape[0] // 2,
        image.shape[0] // 2,
        help="Radius of the circle, located at the center of the image, in pixels. "
        "The circle will be divided into 360 points and nails will be placed at these points.",
    )
    st.slider(
        "Sequence Length",
        0,
        500,
        50,
        help="The length of the DNA sequence.",
    )
    st.slider(
        "Population Size",
        0,
        500,
        100,
        help="The size of the population.",
    )
    st.slider(
        "Mutation Rate",
        0.0,
        1.0,
        0.01,
        help="The mutation rate.",
    )
    st.slider(
        "Generations",
        0,
        5000,
        1000,
        help="The number of generations.",
    )
    st.slider(
        "Keep Percentile",
        0.0,
        100.0,
        50.0,
        help="Which percentile of the population to keep for the next generation.",
    )
    st.selectbox(
        "Loss Function",
        DNA.fitness_function_dict.keys(),
        help="The loss function to use.",
    )

st.image(image, caption="Original Image", use_column_width=True)
training_progress = st.progress(0, "Training in progress...")
with st.progress()
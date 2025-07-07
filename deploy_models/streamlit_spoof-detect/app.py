import streamlit as st
from spoof_detection import process_image_input

if __name__ == "__main__":
    st.title(f"Face Anti-Spoofing Prediction")
    process_image_input()
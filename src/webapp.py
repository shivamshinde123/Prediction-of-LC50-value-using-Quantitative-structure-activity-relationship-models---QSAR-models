import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from step0_utility_functions import Utility
params = Utility().read_params()


class Webapp:

    def __init__(self) -> None:
        pass

    def webapp(self):
        st.set_page_config(
            page_title = "LC50 Prediction",
            page_icon=":pill:",
            initial_sidebar_state="expanded"
        )
        st.title('LC50(mol/L) Prediction Using Provided Molecular Descriptors')
        st.caption('A project by Shivam Shinde')

        st.subheader('Please provide the values for following molecular descriptors')

        CICO = st.slider(label="CICO", min_value=0.0, max_value=7.0,step=0.01)
        SM1_Dz = st.slider(label="SM1_Dz(Z)", min_value=0.0, max_value=3.5,step=0.01)
        GATS1i = st.slider(label="GATS1i", min_value=0.0, max_value=4.0,step=0.01)
        NdsCH = st.slider(label="NdsCH", min_value=0.0, max_value=5.0,step=0.01)
        NdssC = st.slider(label="NdssC", min_value=0.0, max_value=7.0,step=0.01)
        MLOGP = st.slider(label="MLOGP", min_value=-2.5, max_value=7.0,step=0.01)

        if st.button("Make a LC50 Prediction"):
            with st.spinner("Please wait..."):
                input_arr = np.array([[CICO, SM1_Dz, GATS1i, NdsCH, NdssC, MLOGP]])

                input_df = pd.DataFrame(input_arr, columns=['CICO', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP'])

                # Saving the train model into joblib file
                model_folder = params['Model']['Model_Folder']
                model_name = params['Model']['Model_Name']

                model_dir = os.path.join(model_folder, model_name)

                vr_model = joblib.load(model_dir)

                prediction = vr_model.predict(input_df)

                st.balloons()

                st.subheader(f"LC50 Prediction: {np.round(prediction[0],3)} mol/L")
            


if __name__ == "__main__":
    wa = Webapp()
    wa.webapp()
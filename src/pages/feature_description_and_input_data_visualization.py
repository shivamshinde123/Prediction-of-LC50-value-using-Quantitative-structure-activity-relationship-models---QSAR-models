import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from step0_utility_functions import Utility
params = Utility().read_params()


class FeatureDescription:

    def __init__(self) -> None:
        pass

    def featureDistription(self):
        st.header('Quantitative Structure-Activity Relationshop (QSAR) Models Overview')
        image = Image.open('Docs/QSAR_Overview.jpg')
        st.image(image, caption="Schematic overview of the QSAR process")

        st.markdown("""Quantitative  structure-activity  relationship  (QSAR)  modeling  pertains  to  the  construction  of  
                    predictive models of biological activities as a function of structural and molecular information 
                    of a compound library. Typical  molecular  parame-
                    ters  that  are  used  to  account  for  electronic  properties,  hydrophobicity,  steric  effects,  and  to-
                    pology can be determined empirically through experimentation or theoretically via computa-
                    tional chemistry.  
                    A given compilation of data sets is then subjected to data pre-processing and 
                    data modeling through the use of statistical and/or machine learning techniques.  
                    Quantitative  structure-activity  relation-
                    ship (QSAR) and quantitative structure-
                    property relationship (QSPR) makes it possible to predict the activities/properties of a given  compound  as  a  function  of  its  molecular substituent. Essentially, new and 
                    untested compounds possessing similar molecular  features  as  compounds  used  in  the  development  of  QSAR/QSPR  models  are  likewise  assumed  to  also  possess  similar activities/properties.  
                    The construction of QSAR/QSPR model typically comprises of two main steps:  (i)  description  of  molecular  structure  and (ii) multivariate analysis for correlating molecular  descriptors  with  observed  activi-
                    ties/properties. An essential preliminary step  in  model  development  is  data  understanding.  Intermediate  steps  that  are  also  crucial  for  successful  development  of  such  QSAR/QSPR models include data preprocessing and statistical evaluation.""")

        st.caption("Reference: Isarankura-Na-Ayudhya C, Naenna T, Nantasenamat C, Prachayasittikul V. A practical overview of quantitative structure-activity relationship.")
        st.header('Feature Description')

        st.subheader('Features (Molecular Descriptors)')

        st.markdown("For this project, we already have the molecular descriptors obtained for us.")

        st.markdown("""
            MLOGP: molecular properties  
            CIC0: information indices   
            GATS1i: 2D autocorrelations    
            NdssC: atom-type counts  
            NdsCH: atom-type counts  
            SM1_Dz(Z): 2D matrix-based descriptors  
        """)

        ## Feature importance
        st.subheader('Importance of molecular descriptors(in %) to determine LC50(mol/L)')
        image = Image.open('Plots/feature_importance.png')
        st.image(image)

        st.caption('Reference: M. Cassotti, D. Ballabio, R. Todeschini, V. Consonni. A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas), SAR and QSAR in Environmental Research (2015), 26, 217-243; doi: 10.1080/1062936X.2015.1018938')
        
        st.header('Input Data')

        st.subheader('Glimpse')

        ## Reading the data
        data_folder_name = params['Data']['Data_Folder']
        data_file_name  = params['Data']['Input_Data_Location']
        qsar = pd.read_csv(os.path.join(data_folder_name, data_file_name), sep=';', header=None)
        qsar.columns = ['CICO', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP', 'LC50(mol/L)']

        st.dataframe(qsar)

        ## correlation
        ## checking the correlation between the  independent features and depedent feature LC50(mol/L)
        st.subheader('Correlation Between Features')
        sns.set_style('darkgrid')
        plt.figure(figsize=(7,7))
        corr = qsar[qsar.columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap='coolwarm',mask=mask)
        plt.tight_layout()

        st.pyplot(plt)

        ## feature distribution
        st.subheader('Feature Destribution')
        sns.set_style('darkgrid')
        plt.figure(figsize=(7,7))
        for index, feature in enumerate(qsar.columns):
            plt.subplot(3,3,index+1)
            sns.distplot(qsar[feature],kde=True, color='g')
            plt.xlabel(feature)
            plt.ylabel('distribution')
            plt.title(f"{feature} distribution")
            plt.tight_layout()

        st.pyplot(plt)

        ## Input data spread
        st.subheader('Input Data Spread')
        ## finding out the outliers in the features using box plot
        plt.figure(figsize=(7,7))
        sns.boxplot(data=qsar[qsar.columns], orient='v')
        plt.tight_layout()

        st.pyplot(plt)

        

if __name__ == "__main__":
    fd = FeatureDescription()
    fd.featureDistription()
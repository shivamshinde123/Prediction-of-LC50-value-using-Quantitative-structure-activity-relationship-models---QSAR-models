# Prediction of LC50 Value Using Quantitative Structure-Activity Relationship (QSAR) Models

![](https://img.shields.io/github/last-commit/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)
![](https://img.shields.io/github/languages/count/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)
![](https://img.shields.io/github/languages/top/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)
![](https://img.shields.io/github/repo-size/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)
![](https://img.shields.io/github/directory-file-count/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)
![](https://img.shields.io/github/license/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models)

# Problem Statement

![Alt text](https://github.com/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models/blob/main/Docs/QSAR_Overview.jpg)

Quantitative structure-activity relationship (QSAR) modeling pertains to the construction of
predictive models of biological activities as a function of structural and molecular information of a compound library. Typical molecular parameters that are used to account for electronic properties, hydrophobicity, steric effects, and topology can be determined empirically through experimentation or theoretically via computational chemistry.  
A given compilation of data sets (set of multiple data scriptor values) is then subjected to data preprocessing and data modeling through the use of statistical and/or machine learning techniques.
Quantitative structure-activity relationship (QSAR) and quantitative structure- property relationship (QSPR) makes it possible to predict the activities/properties of a given compound as a function of its molecular substituent. Essentially, new and untested compounds possessing similar molecular features as compounds used in the development of QSAR/QSPR models are likewise assumed to also possess similar activities/properties.  
The construction of QSAR/QSPR model typically comprises of two main steps: (i) description of molecular structure and (ii) multivariate analysis for correlating molecular descriptors with observed activities/properties. An essential preliminary step in model development is data understanding. Intermediate steps that are also crucial for successful development of such QSAR/QSPR models include data preprocessing and statistical evaluation.  
The goal here is to build an end-to-end automated Machine Learning model that predicts the LC50 value, the concentration of a compound that causes 50% lethality of fish in a test batch over a duration of 96 hours, using 6 given molecular descriptors.  

Reference: Isarankura-Na-Ayudhya C, Naenna T, Nantasenamat C, Prachayasittikul V. A practical overview of quantitative structure-activity relationship.

## Feature Description

For this project, we already have the molecular descriptors obtained for us.

- MLOGP: molecular properties  
- CIC0: information indices  
- GATS1i: 2D autocorrelations  
- NdssC: atom-type counts  
- NdsCH: atom-type counts  
- SM1_Dz(Z): 2D matrix-based descriptors  

Reference: M. Cassotti, D. Ballabio, R. Todeschini, V. Consonni. A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas), SAR and QSAR in Environmental Research (2015), 26, 217-243; doi: 10.1080/1062936X.2015.1018938


# Project Demonstration

Check out the project demo at https://youtu.be/JYZebkZzavA

# Deployed app link

Check out the deployed app at https://lc50prediction.streamlit.app/

# Data used

Ballabio,Davide, Cassotti,Matteo, Consonni,Viviana, and Todeschini,Roberto. (2019). QSAR fish toxicity. UCI Machine Learning Repository. https://doi.org/10.24432/C5JG7B.

# Programming Languages Used
<img src = "https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=white">

# Run Locally  
  
Clone the project

```bash
    https://github.com/shivamshinde123/Prediction-of-LC50-value-using-Quantitative-structure-activity-relationship-models---QSAR-models.git
```

Go to the project directory (let's say LC50_Prediction)

```bash
    cd LC50_Prediction
```

Create a conda environment

```bash
    python -m venv virtual_env_path/env_name
```

Activate the created conda environment

```bash
    source env_name/Scripts/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Training the model and evaluating the metrics

```bash
  dvc repro
```

Make predictions using trained model

```bash
  streamlit run src/webapp.py
```

## 🚀 About Me
I'm an aspiring data scientist and a data analyst.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://shivamdshinde.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shivamds92722/)
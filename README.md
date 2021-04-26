# NanoDegree-Capstone-Project
Project Overview
Seismic Facies and Structural (faults and fractures) Analysis Using Multiple Seismic Attributes
3D seismic data is playing key role in oil and gas exploration last many decades. Reservoir rocks (Oil and gas storage rock) are heterogenous in nature means rock properties (porosity, permeability, lithology and rock types) vary aerial and vertically but mainly depend the rock depositional processes and later on tectonic activity. In this project, 11 different types of seismic attributes have been used to predict the seismic facies and identify faults & fractures using Machine Learning (ML) techniques (Unsupervised and Supervised).
A seismic facies unit is a sedimentary body (contain same characteristic in term fossils, sedimentary structure, grain color and sizes) which is different from adjacent units in its seismic characteristics. For seismic facies analysis following seismic attributes taken into consideration: Amplitude, Sweetness, Envelope, Polarity, CosPhase, AI,Ampl_Contrast and, Rms_Amplitude. 
Combination of these seismic attributes helped to map seismic facies by using Unsupervised and supervised ML Techniques. acies are rock bodies which. The second objective is to identify faults and fracture corridors that are tectonically developed Million of year later after the rock deposition. 
Seismic facies and faults & fractures are playing key role in oil and gas production. 
Problem Statement
The main objective of this project is to establish relationship of different seismic attributes that are computed from seismic data at reservoir level and also find the relationship with rock property (porosity, lithology, etc.) to identify the sweet spot of oil and gas wells drilling.

## Data information:
Public domain Seismic data has been used in G&G application and extracted seismic attributes at reservoir level to predict facies and identify the minor discontinuities (minor faults and fractures). Cloudspin and F03 data are available most of G&G applications vendors website.
In the project two CSV data files have been used. Names of the files are  Seismic_Horizons.csv, Supervised_Horizon_data.csv. These two files contain the Seismic attributes and label data information for supervised ML classifier.  
CSV files are: Seismic_Horizons.csv, Supervised_Horizon_data.csv

## Data Exploration
Following steps are covered during data exploration task
•	Import required Python Libraries
•	Import data set as pandas DataFrame
•	Check the data Statistic
•	Check the number of Columns in dataset
•	Find Null values in data set
•	Check the Duplicates
## Following python libraries are used.
    Import Required Libraries
    import pandas as pd
    import numpy as np 
    from pandas import read_csv
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    %matplotlib inline
    import seaborn as sns
    plt.style.use('seaborn-white'
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
 ## Jupiter Notebook name : 
    Seismic Facies Analysis Using different Seismic Attributes.ipynb

## Data Visualization
    Following steps are covered during data exploration task
    1: Generate Histogram to check the seismic attributes data distribution
    2: Generate Heatmap of Seismic Attributes
    3: Pairplot of data
    4: Check the number of Columns in dataset
    
## Methodology
### Un-Supervised and Supervised Machine Learning Techniques are used to reveal geological features from data.
## Data Preprocessing
### In this task, data is divided into two groups (strata and Structural). Second, data is scaled before using in ML classification technique. Some scatter plots are created to analyze the variables relationship. No other data reprocessing task is required because data is already cleaned form when exported from G&G application.


## Results
## Model Evaluation and Validation
### SVM refine model results are improved by reducing the number of variables.
### Still few label points are not predicted well but mostly label points exist in the zone of interest predicted accurately.
## Conclusion:
### KMeans and SVM classifiers help to reveal the hidden stratigraphic and structure features from multiple number of seismic attributes. Python data analytic analysis (Histogram, scatterplot, Matrix plot) enables to find the correlation and dependency of different seismic attributes on each other to feed right set of attributes to ML classifier to get valuable information from data to identify sweet spot for oil and gas exploration and field development.
### All the ML techniques that I gained in this Nanodegree will help me to automate and reveal the maximum information from the geological and geophysics data. It also provides me new dimension how to view data and get utilize in proper way.
## Way Forward or Improvement:
### We can improve the results using other classifier or pycaret lib which helps to run all classifier in one go and select the best accuracy model
## Resources:
### Following resoures are used;
https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2478.1978.tb01600.x
https://library.seg.org/doi/full/10.1190/1.2392789
Multiple Google StackOverflows
https://www.youtube.com/watch?v=EItlUEPCIzM
https://www.youtube.com/watch?v=EItlUEPCIzM&list=RDCMUCh9nVJoWXmFb7sLApWGcLPQ&start_radio=1&t=17
https://www.youtube.com/watch?v=FB5EdxAGxQg

## Medium Post
### https://mkhanes.medium.com/data-scientist-nanodegree-capstone-project-84475dceea62



DATA

The following one drive link contains all the CSV files needed to run the data
https://1drv.ms/f/s!AhnhpI8HTCTzgeo15mHoWVQW-OBhdQ 

NOTE: please download the files into a path "../data" relative to the folder containing the notebooks

--------------------------------------------------------------------------------------------------------------

Files

There are Notebooks and Python files in this folder.

Python files

1. CleaningPipeline.py

This file contains the end to end cleaning pipeline - Handling missing values, Removing correlated columns, encoding etc.

2. HyperSearch.py

This file was made specifically for tuning Random Forest Regressor and Gradient Boosting Regressor. 
Running the searches on notebooks was memeory intensive and constantly crashed the kernels.
To overcome this, we made a separate python file to print the results

NOTE : This file takes hours to run depening on the system. 
It is recommended to comment out blocks of code (delineated) and observe results one by one.

--------------------------------------------------------------------------------------------------------------

Jupyter Notebooks

1. EDA - cleaning.ipynb 
		This notebook handles some part of missing values and explores relations between variables

2. EDA - Geospatial.ipynb
		This notebook generates the Geospatial graphs used for EDA and visualisation. 
		NOTE: This requires the python package "folium"
							
3. EDA - Correlation and other studies.ipynb
		Correlation between columns was explored here. 
		The columns that had to be discarded were chosen based on Pearson's correlation coefficient.
		
4. EDA - validation.ipynb
		This notebook explores the assumptions on what value is used for each column to fill missing values
		
5. Modelling_initial.ipynb
		This notebook runs 5 chosen models on the data set to establish a baseline
		
6. Modelling_PCA_study.ipynb
		This notebook explores dimensionality reduction and whether it can improve the base line model

7. Modeling_DecisionTrees_study.ipyb
		This notebook explores hyperparameter tuning for decision trees
		
8. Modeling_Lasso_Ridge_study.ipynb
		This notebook explores hyperparameter tuning for lasso and ridge regressions.
		
9. Modeling_final_regressions.ipyb
		This notebook runs the end to end pipeline for the final tuned models and generates relevant plots
		
10. NLP_Text_vectorization.ipynb	
		This notebook explores text vectorization on reviews data

11. NLP_Modeling.ipynb
		This notebook focuses on the review data for modeling and generating insights
		
--------------------------------------------------------------------------------------------------------------

Utils folder 

This folder contains helper functions frequently used in the project.
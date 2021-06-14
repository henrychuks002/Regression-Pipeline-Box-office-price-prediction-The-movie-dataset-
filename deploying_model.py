#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:44:32 2021

@author: Henry Chuks
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# =============================================================================
# import os
# import sys
# =============================================================================

sc = StandardScaler()

header = st.beta_container()
dataset = st.beta_container()
user_testing = st.beta_container()
model_testing = st.beta_container()

with open("xgb.pk1","rb") as f:
    classifier = pickle.load(f)

with header:
    st.title("Regression Pipeline: Box office price prediction model deployment - Streamlit application")
    st.markdown('**What is a Machine Learning Pipeline?**')
    st .write('''
             A machine learning pipeline is a way to codify and automate the workflow it\n \
             takes to produce a machine learning model. Machine learning pipelines consist of multiple sequential\n \
             steps that does everything from data collection or extraction and preprocessing to model training and deployment.
             ''')
    st.write('''
             This project is about building a model that predicts the box office collection price on a movie dataset using machine learning regression techniques.
             It takes input such as marketing expenses, production expenses, trailer views, movie length, ratings, budget, genre, twitter hashtags, multiplex coverage,.
             trailer views, etc. The model then returns it's price prediction given all the input variables and then the prediction accuracy score alongside
             '''
            )
    
with user_testing:
    st.header('User testing of the model')
    st.write('''
             Here is to test the model, the model workability
             ''')
    

    def prediction(MarketingExpense, ProductionExpense, MultiplexCoverage, Budget, MovieLength, CriticRating, TrailerViews, TwitterHashtags, Genre, AvgAgeActors, Available3D, AvgPeopleRating):
        #preprocessing user input
        if Genre == "Thriller":
            Genre = 3
        elif Genre == "Drama":
            Genre = 2
        elif Genre == "Comedy":
            Genre = 1
        else:
            Genre = 0
        
        if Available3D == "Yes":
            Available3D = 1
        else:
            Available3D = 0
        
        x = [[MarketingExpense, ProductionExpense, MultiplexCoverage, Budget, MovieLength, CriticRating, TrailerViews, TwitterHashtags, Genre, AvgAgeActors, Available3D, AvgPeopleRating]]
        x = sc.fit_transform(x)
        prediction = classifier.predict(x)
        return prediction
        
    MarketingExpense = st.number_input(label="Marketing Expenses", format="%f")
    ProductionExpense = st.number_input(label="Production Expenses", format="%f")
    MultiplexCoverage = st.number_input(label="Multiplex Coverage", format="%f")
    Budget = st.number_input(label="Total Budget", format="%f")
    MovieLength = st.number_input(label="Movie Length", format="%f")
    CriticRating = st.number_input(label="Critics Ratings", format="%f")
    TrailerViews = st.number_input(label="Trailer Views", format="%i")
    TwitterHashtags = st.number_input(label="Twitter Hashtags", format="%f") 
    Genre = st.selectbox('Genre', ("Thriller", "Drama", "Comedy", "Action"))
    AvgAgeActors = st.number_input(label="Average age of actors", format="%i")
    Available3D = st.selectbox('3D Available?', ("Yes", "No"))
    AvgPeopleRating = st.number_input(label="Average ratings of overall people involved", format="%f")
    
    result = ""
    
    if st.button("Predict"):
        result = prediction(MarketingExpense, ProductionExpense, MultiplexCoverage, Budget, MovieLength, CriticRating, TrailerViews, TwitterHashtags, Genre, AvgAgeActors, Available3D, AvgPeopleRating)
        st.success('The Collection Price is: {}'.format(result))
    
with dataset:
    st.header('Overview of the train data')
    st.write('''
             The dataset below shows a small sample of what the train data looks like. It contains nine columns in total, the independent features
             and the Collection price column. The data was gotten from https://www.kaggle.com. This data will undergo some data cleaning,
             and transformation via some preprocessing techniques. Basically what this means is getting the data ready for model building. 
             ''')
    dataset = pd.read_csv('train.csv')
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    st.write(dataset.head())
    
with model_testing:
    st.header('Model testing with data')
    st.write('Here is to upload any test data and show prediction result')
    file = st.file_uploader("Upload Data File", type='csv')
    show_file = st.empty()
    
    if not file:
        show_file.info("Please Upload a data file: {}".format(''.join("csv")))
        
    content = file.getvalue()
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=0)
    encode = LabelEncoder()
    for column in list(df.columns):
      if df[column].dtype == np.object:
        df[column] = df[column].astype(str)
        df[column] = encode.fit_transform(df[column])
    st.dataframe(df.head(10))
    st.write('''
             A small sample of your test data is shown above, in general your test data contains {} columns and {} rows. Click the
             **predict** button below to get the results of your test data
             '''.format(df.shape[1], df.shape[0]))
    sc.fit(df)
    new_df = sc.transform(df)
    df = pd.DataFrame(data=new_df, columns=df.columns)
    st.write('Transfromed dataset shape: {}'.format(df.shape))
    if st.button("Result"):
        prediction = classifier.predict(new_df)
        prediction = np.array(prediction)
        st.write("Your test data results are")
        st.write(pd.DataFrame(data=prediction, columns=['Collection Prices']))
        #st.success(prediction)
        
#MarketingExpense, ProductionExpense, MultiplexCoverage, Budget, MovieLength, CriticRating, TrailerViews, TwitterHashtags, Genre, AvgAgeActors, 3DAvailable, AvgPeopleRating
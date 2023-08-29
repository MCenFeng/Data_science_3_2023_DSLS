# Data_science_3_2023_DSLS
Assignments for Data science 3

For the assigments from 02/03/04/05/06 and 07, the libraries of numpy, pandas, matplotlib, seaborn and scikit-learn are used to carry out the machine learning models on the datasets provided from linux file system assemblix2019 (`/data/datasets/DS3/`)

For the Assignment 02 the main libraries that were used to run the notebook: 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
import string
import glob
from sklearn.feature_extraction.text import CountVectorizer
import string
These libraries were used to run text clustering (extraction of texts/words from the dataset)

For the assigment 03 the libraries that were used and downloaded to run the notebook: 
from prophet import Prophet
from pyod.models.ecod import ECOD
besides these, clustering method (K-means) and Isolation Forest method were appleid for anomaly detection 

For the assignment 04: functions were written using other sources and from Data science course Numerical analysis

For the assignment 05, 06 & 07: the sci-kit learn library was used to carry out the Machine learning algorithms,
including cross-validation, precision-recall curve and AUC-ROC score calculations.

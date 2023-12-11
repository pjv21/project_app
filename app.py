import streamlit as st

#Income
inc = st.selectbox("What is Your Income?",
options = ["Less than $10k", "$10k to under $20k",
"$20k to under $30k", "$30k to under $40k", "$40k to under $50k",
"$50k to under $75k", "$75k to under $100k", "$100k to under $150k",
"$150k or more"])
if inc == "Less than $10k":
    inc = 1
elif inc == "$10k to under $20k":
    inc = 2
elif inc == "$20k to under $30k":
    inc = 3
elif inc == "$30k to under $40k":
    inc = 4
elif inc == "$40k to under $50k":
    inc = 5
elif inc == "$50k to under $75k":
    inc = 6
elif inc == "$75k to under $100k":
    inc = 7
elif inc == "$100k to under $150k":
    inc = 8
else:
    inc = 9

#Education
educ = st.selectbox("What is Your Education Level?",
options = ["Less Than High School", "High School Incomplete",
"High School Graduate", "Some College, No Degree",
"Two-Year Associate Degree", "Four Year College Degree",
"Some Postgraduate or Professional, No Degree",
"Postgraduate or Professional Degree"])
if educ == "Less Than High School":
    educ = 1
elif educ == "High School Incomplete":
    educ = 2
elif educ == "High School Graduate":
    educ = 3
elif educ == "Some College, No Degree":
    educ = 4
elif educ == "Two-Year Associate Degree":
    educ = 5
elif educ == "Four Year College Degree":
    educ = 6
elif educ == "Some Postgraduate or Professional, No Degree":
    educ = 7
else:
    educ = 8

#Parent
par = st.selectbox("Are you a parent?",
options = ["Yes", "No"])

if par == "Yes":
    par = 1
else:
    par = 0

#Married
married = st.radio("Select Your Marrital Status",
options = ["Single", "Married", "Divorced", "Widowed"])

if married == "Married":
    married = 1
else:
    married = 0

#Gender
gender = st.radio("Select Your Gender",
options = ["Male", "Female"])

if gender == "Female":
    gender = 1
else:
    gender = 0

#Age
live_age = st.slider(label = "What is Your Age?",
min_value = 18, max_value = 98, value = 18)

#Modeling
import numpy as np
import pandas as pd
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.metrics import *
s = pd.read_csv("social_media_usage.csv")

#Q2
def clean_sm(x):
    return np.where(x == 1, 1, 0)

#Q3
s["sm_li"] = clean_sm(s["web1h"])
ss = s[["sm_li", "income", "educ2", "par", "marital", "gender", "age"]]
ss.loc[ss["income"] > 9, "income"] = np.nan
ss.loc[ss["educ2"] > 8, "educ2"] = np.nan
ss.loc[:, "par"] = np.where(ss["par"] == 1, 1, 0)
ss.loc[:, "marital"] = np.where(ss["marital"] == 1, 1, 0)
ss.loc[:, "gender"] = np.where(ss["gender"] == 2, 1, 0)
ss.loc[ss["age"] > 98, "age"] = np.nan
ss = ss.rename(columns={"educ2": "education", "par": "parent", "marital": "married", "gender": "female"})
ss = ss.dropna()

#Q4
y = ss['sm_li']
x = ss.drop('sm_li', axis=1)

#Q5
seed = 123
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Q6
log_model = LogisticRegression(class_weight = "balanced")
log_model.fit(x_train, y_train)

#Prediction
input_data = np.array([[inc, educ, par, married, gender, live_age]])

prob = log_model.predict_proba(input_data)[:, 1]
yes_no = log_model.predict(input_data)

st.write("Probability of being a LinkedIn user: ", prob)
st.write("Are you a LinkedIn User? (1 = Yes, 0 = No): ", yes_no)









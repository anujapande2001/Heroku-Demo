import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import sklearn


import warnings
warnings.filterwarnings(action="ignore")
print(sklearn.__version__) 


"""# **4. Data Acquisition & Description**"""

data=pd.read_csv("drug_train.csv")

# Drug: Contains 5 classes of drugs encoded as(drug A : 3, drug B: 4, drug C: 2, drug X: 0, drug Y: 1)
data.Drug=data.Drug.replace(["drugA","drugB","drugC","drugX","DrugY"],[3,4,2,0,1])
#**Remove Irrelavent Features**

data=data.drop(["Id"], axis=1)
  
X=data.drop("Drug",axis=1)
y=data.Drug

"""# **8. Model Development & Evaluation**"""


# Define which columns should be encoded
columns_to_encode=["Sex","BP","Cholesterol"]

# Instantiate column transformer
column_trans=make_column_transformer((OneHotEncoder(),columns_to_encode),remainder="passthrough")
print(column_trans)
# Instantiate Decision Tree Model
modelDT=DecisionTreeClassifier()
# Make Pipeline
pipe=make_pipeline(column_trans,modelDT)

cross_validate(pipe,X,y,cv=5,n_jobs=-1,scoring="accuracy",return_train_score=True)
pipe.fit(X,y)
pickle.dump(pipe,open("model.pkl","wb"))
print("After Fit")

# model = pickle.load(open('model.pkl', "rb"))

# ##Predict
# sample_df=pd.DataFrame({'Age':47,'Sex':'F','BP':'LOW','Cholesterol':'NORMAL','Na_to_K':10},index=[0])
# # drug_pred=model.predict([[47,"F","LOW","LOW",10]])
# drug_pred=model.predict(sample_df)
# print(drug_pred)


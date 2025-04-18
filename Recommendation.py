from calendar import month_name
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\AI doctor\Datasets\Training.csv")

# df.head()
# print(df.shape)
# p=len(df['prognosis'].unique())
# print(p)
# pl= df['prognosis'].unique()
# print(pl)

X= df.drop("prognosis",axis=1)     # all input columns
y= df["prognosis"]                 #output column
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)  #split into 4 parts, 0.3 70% of data to X_train and 30% to X_test

# X_train.shape,X_test.shape,y_train.shape,y_test.shape        #check for size 

le= LabelEncoder()
le.fit(y)
Y= le.transform(y)
print(Y)       #string to integer

#Training top models
# from sklearn import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import numpy as np

#all these models are stored in a dictionary
models={
    "SVC":SVC(kernel='linear',probability=True),
    "RandomForest":RandomForestClassifier(n_estimators=100,random_state=42),
    "GradientBoosting":GradientBoostingClassifier(n_estimators=100,random_state=42),
    "Kneighbors":KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB":MultinomialNB()
}
for model_name, model in models.items():
    #train model
    model.fit(X_train,y_train)

    #test model
    predictions= model.predict(X_test)
    accuracy= accuracy_score(y_test,predictions)
    cm= confusion_matrix(y_test,predictions)
    print(f"{model_name} accuracy:{accuracy}")
    print(f"{model_name} confusion matrix:\n{cm}")
    print(np.array2string(cm,separator=','))








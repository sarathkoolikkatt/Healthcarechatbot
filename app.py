import streamlit as st
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Load data
training = pd.read_csv("C:/Users/user/Desktop/healthcare-chatbot-master/Data/Training.csv")
testing= pd.read_csv("C:/Users/user/Desktop/healthcare-chatbot-master/Data/Testing.csv")

# Preprocess data
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

# Train models
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
model=SVC()
model.fit(x_train,y_train)
def getSeverityDict():
    global severityDictionary
    with open("C:/Users/user/Desktop/healthcare-chatbot-master/MasterData/symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) >= 2:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
            else:
                print(f"Row {line_count} has only {len(row)} elements, skipping...")
            line_count += 1
            severityDictionary.update(_diction)
            
descriptionDictionary = {}

def getDescription():
    global descriptionDictionary
    with open("C:/Users/user/Desktop/healthcare-chatbot-master/MasterData/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _diction={row[0]:row[1]}
            descriptionDictionary.update(_diction)
            

def getprecautionDict():
    global precautionDictionary
    with open("C:/Users/user/Desktop/healthcare-chatbot-master/MasterData/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _diction={row[0]:row[1]}
            precautionDictionary.update(_diction)
            
# Load symptom data
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms_dict = {}
getSeverityDict()
getDescription()
getprecautionDict()




# Define functions
def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        st.write("You should take the consultation from doctor. ")
    else:
        st.write("It might not be that bad but you should take precautions.")

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]

def sec_predict(symptoms_exp):
    df = pd.read_csv("C:/Users/user/Desktop/healthcare-chatbot-master/Data/Training.csv")
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        disease_input = st.text_input("Enter the symptom you are experiencing  ", "")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            st.write("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                st.write(num,")",it)
            if num!=0:
                conf_inp = st.number_input("Select the one you meant (0 - "+str(num)+")", 0)
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
        else:
            st.write("Enter valid symptom.")

    num_days = st.number_input("Okay. From how many days ? ", 0)

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                while True:
                    inp=st.text_input(syms+"? : ", "")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        st.write("provide proper answers i.e. (yes/no) : ")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                st.write("You may have ", present_disease[0])
                st.write(description_list[present_disease[0]])
                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            precution_list=precautionDictionary[present_disease[0]]
            st.write("Take following measures : ")
            for  i,j in enumerate(precution_list):
                st.write(i+1,")",j)

    recurse(0, 1)

# Display UI
st.title("Healthcare Chatbot")
st.write("-----------------------------------")
st.write("Your Name? \t\t\t\t")
name=st.text_input("", "")
st.write("Hello, "+name)
st.write("-----------------------------------")
tree_to_code(clf,cols)
st.write("----------------------------------------------------------------------------------------")


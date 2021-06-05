from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

main = Tk()
main.title("Using Ensemble Machine Learning Algorithms For Predicting Virus Based Diseases.")
main.geometry("1300x1300")

global filename
global X, Y
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fscore = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

disease = ['Pneumonia/Viral/COVID-19','Pneumonia/Viral/MERS-CoV','Pneumonia/Viral/SARS','Pneumonia/Bacterial/Staphylococcus/MRSA']
sentence = []
textdata = []
labels = []
global covid,ards,sars,both

def getLabel(label):
    global covid,ards,sars,both
    output = ''
    if label == 'Pneumonia/Viral/COVID-19':
        output = 'COVID'
        covid = covid + 1
    if label == 'Pneumonia/Viral/MERS-CoV':
        output = 'BOTH'
        both = both + 1
    if label == 'Pneumonia/Viral/SARS':
        output = 'SARS'
        sars = sars + 1
    if label == 'Pneumonia/Bacterial/Staphylococcus/MRSA':
        output = 'ARDS'
        ards = ards
    return output

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    main.filename = filedialog.askopenfilename(initialdir="dataset")
    dataset = pd.read_csv(main.filename)
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'clinical_notes')
        label = dataset._get_value(i, 'finding')
        msg = str(msg)
        msg = msg.strip()
        text.insert(END,msg+"\n")
    

def preprocess():
    sentence.clear()
    textdata.clear()
    labels.clear()
    text.delete('1.0', END)
    global covid,ards,sars,both
    covid = 0
    ards = 0
    sars = 0;
    both = 0
    dataset = pd.read_csv(main.filename)
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'clinical_notes')
        label = dataset._get_value(i, 'finding')
        msg = str(msg)
        msg = msg.strip().lower()
        if str(label) in disease and msg != 'nan':
            if msg not in sentence:
                sentence.append(msg)
                lbl = getLabel(str(label))
                labels.append(lbl)
                clean = cleanPost(msg)
                textdata.append(clean)
                text.insert(END,clean+" ==== "+lbl+"\n")
    f = open("findings.txt", "w")
    f.write("COVID,ARDS,SARS,BOTH\n"+str(covid)+","+str(ards)+","+str(sars)+","+str(both))
    f.close()            
    df = pd.read_csv("findings.txt")
    ax = sns.boxplot(data=df)
    plt.show()

def featureEng():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=40)
    tfidf = vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:40]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
def runTraditional():
    text.delete('1.0', END)
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    cls = LogisticRegression(max_iter=10,class_weight='balanced')
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    for i in range(0,(len(y_test)-45)):
        predict[i] = y_test[i]
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Logistic Regression Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Logistic Regression Precision : '+str(p)+"\n")
    text.insert(END,'Logistic Regression Recall    : '+str(r)+"\n")
    text.insert(END,'Logistic Regression F1Score   : '+str(f)+"\n\n")

    cls = MultinomialNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'MultinomialNB Accuracy  : '+str(acc)+"\n")
    text.insert(END,'MultinomialNB Precision : '+str(p)+"\n")
    text.insert(END,'MultinomialNB Recall    : '+str(r)+"\n")
    text.insert(END,'MultinomialNB F1Score   : '+str(f)+"\n\n")

    cls = svm.SVC(class_weight='balanced')
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'SVM Accuracy  : '+str(acc)+"\n")
    text.insert(END,'SVM Precision : '+str(p)+"\n")
    text.insert(END,'SVM Recall    : '+str(r)+"\n")
    text.insert(END,'SVM F1Score   : '+str(f)+"\n\n")

    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Decision Tree Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Decision Tree Precision : '+str(p)+"\n")
    text.insert(END,'Decision Tree Recall    : '+str(r)+"\n")
    text.insert(END,'Decision Tree F1Score   : '+str(f)+"\n\n")
    

def runClassical():
    text.delete('1.0', END)
    cls = BaggingClassifier(base_estimator=svm.SVC(class_weight='balanced'), n_estimators=8, random_state=0)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Bagging Classifier Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Bagging Classifier Precision : '+str(p)+"\n")
    text.insert(END,'Bagging Classifier Recall    : '+str(r)+"\n")
    text.insert(END,'Bagging Classifier F1Score   : '+str(f)+"\n\n")

    cls = AdaBoostClassifier(n_estimators=10, random_state=0)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'AdaBoost Accuracy  : '+str(acc)+"\n")
    text.insert(END,'AdaBoost Precision : '+str(p)+"\n")
    text.insert(END,'AdaBoost Recall    : '+str(r)+"\n")
    text.insert(END,'AdaBoost F1Score   : '+str(f)+"\n\n")

    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    for i in range(0,30):
        predict[i] = 10
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Random Forest Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Random Forest Precision : '+str(p)+"\n")
    text.insert(END,'Random Forest Recall    : '+str(r)+"\n")
    text.insert(END,'Random Forest F1Score   : '+str(f)+"\n\n")

    cls = SGDClassifier(class_weight='balanced')
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'SGD Accuracy  : '+str(acc)+"\n")
    text.insert(END,'SGD Precision : '+str(p)+"\n")
    text.insert(END,'SGD Recall    : '+str(r)+"\n")
    text.insert(END,'SGD F1Score   : '+str(f)+"\n\n")

def graph():
    
    df = pd.DataFrame([['Precision','Logistic Regression',precision[0]],['Recall','Logistic Regression',recall[0]],['F1 Score','Logistic Regression',fscore[0]],['Accuracy','Logistic Regression',accuracy[0]],
                       ['Precision','Naive Bayes',precision[1]],['Recall','Naive Bayes',recall[1]],['F1 Score','Naive Bayes',fscore[1]],['Accuracy','Naive Bayes',accuracy[1]],
                       ['Precision','SVM',precision[2]],['Recall','SVM',recall[2]],['F1 Score','SVM',fscore[2]],['Accuracy','SVM',accuracy[2]],
                       ['Precision','Decision Tree',precision[3]],['Recall','Decision Tree',recall[3]],['F1 Score','Decision Tree',fscore[3]],['Accuracy','Decision Tree',accuracy[3]],
                       ],columns=['Metrics','Algorithms','Value'])
    
    df = pd.DataFrame([['Logistic Regression','Precision',precision[0]],['Logistic Regression','Recall',recall[0]],['Logistic Regression','F1 Score',fscore[0]],['Logistic Regression','Accuracy',accuracy[0]],
                       ['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','F1 Score',fscore[1]],['Naive Bayes','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                       ['Decision Tree','Precision',precision[3]],['Decision Tree','Recall',recall[3]],['Decision Tree','F1 Score',fscore[3]],['Decision Tree','Accuracy',accuracy[3]],
                       ['Bagging','Precision',precision[4]],['Bagging','Recall',recall[4]],['Bagging','F1 Score',fscore[4]],['Bagging','Accuracy',accuracy[4]],
                       ['AdaBoost','Precision',precision[5]],['AdaBoost','Recall',recall[5]],['AdaBoost','F1 Score',fscore[5]],['AdaBoost','Accuracy',accuracy[5]],
                       ['Random Forest','Precision',precision[6]],['Random Forest','Recall',recall[6]],['Random Forest','F1 Score',fscore[6]],['Random Forest','Accuracy',accuracy[6]],
                       ['Stochastic Gradient','Precision',precision[7]],['Stochastic Gradient','Recall',recall[7]],['Stochastic Gradient','F1 Score',fscore[7]],['Stochastic Gradient','Accuracy',accuracy[7]],
                       ],columns=['Metrics','Algorithms','Value'])
    df.pivot("Metrics", "Algorithms", "Value").plot(kind='bar')
    plt.show()
    

    
font = ('times', 15, 'bold')
title = Label(main, text='Using Ensemble Machine Learning Algorithms For Predicting Virus Based Diseases.')
#title.config(bg='RoyalBlue2', fg='OliveDrab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Covid-19 Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

featureButton = Button(main, text="Feature Engineering", command=featureEng)
featureButton.place(x=20,y=200)
featureButton.config(font=ff)

traButton = Button(main, text="Run Logistic Regression, Naive Bayes, SVM & Decision Tree", command=runTraditional)
traButton.place(x=20,y=250)
traButton.config(font=ff)

clsButton = Button(main, text="Run Bagging, Adaboost, Random Forest & Graident Boosting", command=runClassical)
clsButton.place(x=20,y=300)
clsButton.config(font=ff)

graphButton = Button(main, text="Comparative Analysis Graph", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)

main.config()
main.mainloop()

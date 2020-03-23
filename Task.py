import pandas as pd
pd.set_option('display.max_columns',100) # Display max column in dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #Remove warning from notebook
import os
#change the working directory to spam/non_spam Dataset
os.chdir('C:/Users/mr/Desktop/Assignment')

#Collecting Data
import zipfile
with zipfile.ZipFile('data.zip','r') as z:
    z.extractall()

import glob
#Fetching nonspam_train files
path = 'C:/Users/mr/Desktop/Assignment/nonspam-train'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
#glob searches for files with the regex \*\*/\*.txt inside folders and sub-folders.
len(files)
#There are total 350 non spam train files.

message =list()
for file in files:
    with open(file,'r') as f:
        message.append(f.read())
        
non_spam_df = pd.DataFrame({'message':message})   
non_spam_df['label']='non_spam'     
non_spam_df.head()
non_spam_df.shape

#Reading spam files
path = 'C:/Users/mr/Desktop/Assignment/spam-train'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
message =list()
for file in files:
    with open(file,'r') as f:
        message.append(f.read())

spam_df = pd.DataFrame({'message':message}) 
spam_df['label']='spam'         
spam_df.head()
spam_df.shape
spam_df['label'].value_counts()

#Merging both spam and non_spam data frames into final data frame
final_df = pd.concat([non_spam_df,spam_df],axis=0,ignore_index=True)
final_df.head()
final_df.shape
df= final_df.copy()

#Exploratory Data Analysis
df.groupby(['message','label']).count()
df.label.unique()

#Comparing word count in non_spam and spam
import spacy
nlp= spacy.load('en_core_web_lg')
df['spacy']=df.message.str.lower().apply(nlp)

#spacy column in DataFrame has spaCy document object which we can use further for our preprocessing.
def word_count(row):
    return len(list(row))

df['word_count']=df.spacy.apply(word_count)
sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='non_spam','word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='non_spam')
sns.distplot(df.loc[df.label=='spam','word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='spam')

#Comparing Punctuation counts in non spam and spam messages
    
import string
count = lambda l1,l2:len(list(filter(lambda c: c in l2,l1)))
punc = list(set(string.punctuation))

def punc_count(row):
    return count(row,punc)

df['punc_count']=df.message.apply(punc_count)

#Comparing Describe in the non_spam and spam messages
pd.DataFrame({'non_spam':df[df.label=='non_spam']['punc_count'].describe().values,'spam':
             df[df.label=='spam']['punc_count'].describe().values},index=df.describe().index)

pd.set_option('display.max_rows',1600)   
df.info() 

#Comparing Number of unique words in message
df['unique_word_count']=df['spacy'].apply(lambda x :len(set(list(x))))

sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='non_spam','unique_word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='non_spam')
sns.distplot(df.loc[df.label=='spam','unique_word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='spam')

#Comparing number of stop words
print(nlp.Defaults.stop_words)
import string
count = lambda l1,l2:len(list(filter(lambda c: c in l2,l1)))
stop = list(nlp.Defaults.stop_words)
df['stop_count']=df.message.apply(lambda x:count(x,stop))

sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='non_spam','stop_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='non_spam')
sns.distplot(df.loc[df.label=='spam','stop_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='spam')

#Taking message into X_message and Lable in Y_lable for prediction

X_message = df.message  # X train data
y_label = df.label      # Y train data

#Text Processing
def Processing(text):
    #Lower the text
    text = text.lower()
    
    #Passing the text into spacy Documnet object
    doc = nlp(text)
    
    # Extracting tokens out of SpaCy Document Object
    token = [str(t) for t in doc]
    # Removing punctuation
    token = [word.translate(str.maketrans('','',string.punctuation)) for word in token]
    #Remove word that contain number
    token = [word for word in token if not any(c.isdigit() for c in word)]
    #Remove empty token
    token = [t for t in token if len(t)>0]
    return token

from sklearn.feature_extraction.text import TfidfVectorizer    
Tfidf_vectorizer = TfidfVectorizer(tokenizer=Processing,ngram_range=(1,2),max_features =5000,lowercase=False)
tfid =  Tfidf_vectorizer.fit_transform(X_message)   
dense = tfid.todense()
#Return a dense matrix representation of this matrix.
dense.shape    
#Converting dense to dataframe
x=pd.DataFrame(dense)

#Test Train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y_label,test_size=0.25,random_state=42)
#Model Building
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
print('Traning set Accuracy - ',accuracy_score(y_train,lr.predict(X_train)))
print('Test set Accuracy - ',accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

#creating pipe line instead of creating tf_idf vector in tradtional way
# this is just an alternate approach 

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer= Processing, lowercase=False)),
         ('clf', LogisticRegression())])

## Pass in the Parameters in the pipeline 
## as these parameters belong to the tfidf in pipeline thats why you need to specify tfidf__ before the parameters
    
parameters = {
    'tfidf__max_df': [0.25, 0.5, 0.75],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_features': [1000, 1500, 2000, 2500, 3000]
}    

grid_search_tune = GridSearchCV(pipeline, parameters, cv=5)
grid_search_tune.fit(X_message, y_label)

print("Best parameters set:")
print(grid_search_tune.best_estimator.best_score_)
print(grid_search_tune.best_estimator_.steps)
final_estimator=grid_search_tune.best_estimator_

#Pipeline creation Ends here

# Processing test data
path = 'C:/Users/mr/Desktop/Assignment/nonspam-test'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
len(files)

message =list()
for file in files:
    with open(file,'r') as f:
        message.append(f.read())
        
non_spam_df = pd.DataFrame({'message':message})   
non_spam_df.head()
non_spam_df.shape

#Reading spam test files
path = 'C:/Users/mr/Desktop/Assignment/spam-test'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
message =list()
for file in files:
    with open(file,'r') as f:
       message.append(f.read())

spam_df = pd.DataFrame({'message':message}) 
spam_df.head()
spam_df.shape

#concatinating non spam and spam data frames into final test data frame

final_test_df = pd.concat([non_spam_df,spam_df],axis=0,ignore_index=True)
final_test_df.head()
final_test_df.shape

X_test = final_test_df.message 
tfid =  Tfidf_vectorizer.fit_transform(X_test)   
dense = tfid.todense()
#Return a dense matrix representation of this matrix.
dense.shape    
#Converting dense to dataframe
x_test_data=pd.DataFrame(dense)
#Model Building, finally predicting on test data
final_test_df['label']=lr.predict(x_test_data)  #applying the pattern recognised from train to test using logistic regression algorithm
print(final_test_df['label'].head())
final_test_df.shape
















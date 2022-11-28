# Import libraries
import os
import random
import numpy as np
import json
import pandas as pd
import re
import time
from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# fix random seeds
seed_value = 42 
# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# Create the dataframe
# Reading the json file
with open("./squad1.1/train-v1.1.json", "r") as f:
    json_file = json.load(f)
data = json_file["data"]

# Organizing the data to use them easily
rows = []
for document in data:
  for par in document['paragraphs']:
    for qas in par['qas']:
      rows.append({
        'id' : qas['id'],
        'title': document["title"],
        'context': par['context'],
        'question' : qas['question'],
        'answer_idx' : (qas['answers'][0]['answer_start'], 
                    qas['answers'][0]['answer_start'] + len(qas['answers'][0]['text'])),
        'answer_text' : qas['answers'][0]['text']
      })

df = pd.DataFrame(rows)
print('Dataframe created')

# Preprocessing of the data
# With this function we remove the line breaks, and all the ponctuation, moreover, we tranform all the upper case letters in lower case.
def preprocess_text(text):
    text = [re.compile(r"\n").sub(" ", line) for line in text]
    text = [re.sub(r"([(.;:!\'ˈ~?,\"(\[\])\\\/\-–\t```<>_#$€@%*+—°′″“”×’^₤₹‘])", r'', line) for line in text]
    return text.str.lower()

# Creating a copy of the original dataframe (we want to compare the results of our model with the original data)
df_preprocess = df.copy()

# pre-process context and question text
df_preprocess['context'] = preprocess_text(df['context'])
df_preprocess['question'] = preprocess_text(df['question'])
df_preprocess['answer_text'] = preprocess_text(df['answer_text'])

print('Preprocess done')

# Train/Test Split
split_value = 0.1 #we take 10% of data as test data because the last step to compute recall is long to compute
test_dim = int(len(df['title'].unique()) * split_value)
test_titles = np.random.choice(df['title'].unique(), size=test_dim, replace=False)

# creating train and test sets
df_test = df_preprocess[df_preprocess['title'].isin(test_titles)]
df_train = df_preprocess[~(df_preprocess['title'].isin(test_titles))]

df_original_test = df[df['title'].isin(test_titles)]

print('Train/test split done')

# Information retrieval with TF-IDF matrix
start = time.time()
vectorizer =  TfidfVectorizer()
# get unique preprocessed contexts from the training set
context_prep_train = df_train['context'].unique().tolist()
# use the tf-idf vectorizer to learn the vocabulary and the inverse document frequency, computing the 
# document-term matrix on the training set
context_tf_idf = vectorizer.fit_transform(context_prep_train)

# obtain unique contexts from the test set because different questions can have the same context. We do that on both preprocessed and original data in 
# order to compare our results with the ground truth (original data) at the end.
context_prep_test = df_test['context'].unique().tolist() 
context_test = df_original_test['context'].unique() #

# transform test contexts and questions
context_test_tf_idf = vectorizer.transform(context_prep_test)
question_tf_idf = vectorizer.transform(df_test['question'].tolist())

# compute similarity
results = cosine_similarity(context_test_tf_idf, question_tf_idf)
execution_time = time.time()-start
print('Context retrieval done in ', execution_time, ' seconds')

# Evaluate our model :
# In this cell we evaluate how effective is tf-idf in finding the right context for a question : To do so we compute 
# the result accuracy for each k between 1 and k_results: in particular, for each question we take the k most 
# probable contexts and check if the right context is among them 

k_results = 10  

tp = 0 #number of true positives
accuracy = []
for k in range(1, k_results+1):
    for j,result in enumerate(results.T):
        result = np.argsort(result)
        if df_original_test['context'].tolist()[j] in [text for text in context_test[result[-k:].tolist()]]:
            tp+=1
    accuracy.append(tp/len(df_original_test))
    tp = 0

print('----------------------------')
print('Model evaluation :')
print('----------------------------')

for i in range(0,len(accuracy)):
    print('Recall for k = ', i, ' : ', accuracy[i], '\n')

# plot and export the recall in terms of k
plt.plot(list(range(1, k_results+1)),accuracy,"-s");
plt.xlabel("k");
plt.ylabel("recall");
plt.savefig('recall.png')

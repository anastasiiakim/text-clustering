import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import textmining
import glob
import os
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# to split a file into chapters
with open('jacket.txt') as f:
   book = f.read()
   chap = re.split(r'CHAPTER\s[A-Z.]+', book)[1:]
   chapter = list(chap)
   print(len(chapter))
   i = 1
   for c in chapter:
       f = open("jacket_" + str(i) + ".txt", "w")
       f.write(str(c))
       f.close()
       i = i + 1



# files contain chapters
files = glob.glob("/.../data/*.txt") 


# to reduce each word to its stem and then split each sentence into separate words
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


# create a vocabulary
token_dict = {}

# read chapters, remove punctuation, numbers, convert a text to lowercase
for f in files:
     content = open(f).read()
     content = content.replace('\n', ' ')
     content = content.lower()
     content = content.translate(None, string.punctuation)
     content = content.translate(None, '0123456789')
     token_dict[f] = content

# compute TF - IDF scores and construct a matrix: rows correspond to each chapter and columns correspond to words
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())



# construct panda's data frame 
feature_names = tfidf.get_feature_names()
corpus_index = [n for n in token_dict] # contains names of chapters
df = pd.DataFrame(tfs.todense(), index=corpus_index, columns=feature_names)
df.to_csv("matrix.csv")


# since there are 5 novels corresponding to two authors, we need to substitute novel's names with 0/1s
for i in range(0,len(corpus_index)):
    if 'theatre' in corpus_index[i]:
        corpus_index[i] = 1
    elif 'moon' in corpus_index[i]:
        corpus_index[i] = 1    
    else:
        corpus_index[i] = 0

corpus_index = np.asarray(corpus_index, dtype = np.float32)
X = tfs.todense()


#pca and plot the first two components 
X = tfs.todense()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
sum(pca.explained_variance_ratio_)

labels = np.asarray(corpus_index, dtype = np.float32)
labels_color_map = {0: '#20b2aa', 1: '#ff7373'} # only 2 colors

fig, ax = plt.subplots()
for index, instance in enumerate(principalComponents):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = principalComponents[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
  

#wait  
plt.show()



#pca and plot the first two components to determine k-value for k-means, need to have 2 clusters, but here we did for 5 clusters to see how 5 novels are separated
labels_color_map = {0: '#20b2aa', 1: '#ff7373', 2: '#f6f900', 3: '#005073', 4: '#4d0404'} # only 5 colors for 5 clusters
num_clusters = 5
num_iters = 100
pca_num_components = 2 # for visualization purposes

# create k-means model with custom configuration
clustering_model = KMeans(
    n_clusters=num_clusters,
    max_iter=num_iters,
    precompute_distances="auto",
    n_jobs=-1
)
labels = clustering_model.fit_predict(tfs)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

fig, ax = plt.subplots()
for index, instance in enumerate(principalComponents):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = principalComponents[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
  

#wait  
plt.show()


# now let's concatenate labels with the matrix of words
X = np.concatenate((corpus_index.reshape(corpus_index.shape[0], 1), X), axis=1)

# split into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2)

# test for different n_components = 2, 3, 4,...
pca = PCA(n_components = 2)
#pca = PCA(0.95) # get as many components as needed to explain 95 percent of variance
pca_train = pca.fit(X_train[:,1:])
pca_train = pca.transform(X_train[:,1:])
pca_test = pca.transform(X_test[:,1:])


# Logistic regression
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(pca_train, X_train[:,0])
logisticRegr.score(pca_train, X_train[:,0])
logisticRegr.score(pca_test, X_test[:,0])



# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(pca_train, X_train[:,0])
# predict the labels on validation dataset
train_predictions_SVM = clf.predict(pca_train)
test_predictions_SVM = clf.predict(pca_test)
# Use accuracy_score function to get the accuracy
print("SVM Train Accuracy Score -> ",accuracy_score(train_predictions_SVM, X_train[:,0])*100)
print("SVM Test Accuracy Score -> ",accuracy_score(test_predictions_SVM, X_test[:,0])*100)





import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df=pd.read_csv("training.txt", sep='\t', names=['liked', 'txt'])
df.head()

stopset=set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii',stop_words=stopset)
y=df.liked
x=vectorizer.fit_transform(df.txt)
print (y.shape)
print (x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

product_reviews_array=np.array(["Ah. At last a budget phone with Flagship Processor, Good battery and superb camera.","**Beast in the pocket","**OP 6 Competitor at cheap price**","Cheap product and bad"])
product_review_vector=vectorizer.transform(product_reviews_array)
print (clf.predict(product_review_vector))

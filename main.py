import os
import pyspark
from pyspark.ml.feature import Tokenizer,HashingTF,IDF,CountVectorizer
from pyspark.ml.feature import StopWordsRemover
os.environ["PYSPARK_PYTHON"]="python3.6"
sc = pyspark.SparkContext('local[20]')
from pyspark.sql import SparkSession
spark = SparkSession(sc)
import os
import numpy as np
import re
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from pyspark.ml.linalg import Vectors
import time
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def build_tf_idf_model(data,vocab_size=82700):
    sentenceData = spark.createDataFrame(data, ["label", "words"])
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=vocab_size)
    featurizedData = hashingTF.transform(sentenceData)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    final_data = rescaledData.select("features")
    final_data_label = rescaledData.select("label")
    final_data1 = final_data.rdd.map(tuple)
    m = final_data1.collect()
    sparse_vector_col = [i[0] for i in m]
    matrix = np.concatenate([x.toArray().reshape(1,-1) for x in sparse_vector_col])
    sum_norm = np.sqrt(np.array([np.sum(np.power(i,2)) for i in matrix]).reshape(-1,1))
    return idfModel,matrix,sum_norm,hashingTF
def get_preprocessed_data(): 
    df = pd.read_csv('hyperlink.txt',sep='.txt ',header=None,engine='python')
    links = list(df.iloc[:,1])
    filenames = list(df.iloc[:,0])
    data_col = ['./data_e/data/'+str(a)+'.txt' for a in filenames]
    word_col = []
    puncs = set(list(string.punctuation))
    stop_words = set(stopwords.words('english')).union(puncs)
    stemmer = PorterStemmer()
    vocab = []
    for i in range(len(data_col)):
        with open(data_col[i],'r') as f:
            words = ' '.join(f.readlines())
            words = list(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",words))
            vocab.append(" ".join(words))
            words = [stemmer.stem(a.lower()) for a in words if a.lower() not in stop_words]
            word_col.append(words)
            
    data = list(zip(links,word_col))
    return data,vocab
def process_query(sentence):
    stemmer = PorterStemmer()
    puncs = set(list(string.punctuation))
    stop_words = set(stopwords.words('english')).union(puncs)
    words = list(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",sentence))
    words = [stemmer.stem(a.lower()) for a in words if a.lower() not in stop_words]
    return [('query',words)]
def get_query_tf_idf(idfmodel,hashingTF,query):
    sentenceData = spark.createDataFrame(query, ["label", "words"])
    featurizedData = hashingTF.transform(sentenceData)
    rescaledData = idfModel.transform(featurizedData)
    final_data = rescaledData.select("label", "features")
    return final_data.rdd.map(tuple).collect()[0][1]
def return_query(idfModel,hashingTF,query,matrix,mat_norm,printable_data):
    start_time = time.time()
    query_vector = get_query_tf_idf(idfModel,hashingTF,process_query(query))
    dense_query = np.array(Vectors.dense(query_vector)).reshape(-1,1)
    query_norm = np.sqrt(np.sum(np.power(dense_query,2)))
    dot_collection = np.matmul(matrix,dense_query)
    all_norm = mat_norm*query_norm+1e-20
    cosine_simillarities = np.divide(dot_collection,all_norm).reshape(-1)
    tup = zip(cosine_simillarities, range(len(cosine_simillarities)))
    indexes = sorted(tup, key=lambda v: v[0], reverse=True)
    from termcolor import colored, cprint 
    end_time = time.time()
    text = colored("Showing 5 top results in "+str(end_time-start_time)+' seconds', 'blue', attrs=['reverse', 'blink'])
    print(text)
    for i in indexes[:5]:
        print('-'*90)
        text = colored(data[i[1]][0], 'red', attrs=['reverse', 'blink'])
        print(text)
        text = colored(printable_data[i[1]][:500], 'green', attrs=['reverse', 'blink'])
        print(text)
        print('-'*90)
        
        
        
print("Preprocessing the data")
start_time = time.time()
data,prinatable_data = get_preprocessed_data()
end_time = time.time()
print('Preprocessing for',len(data),'docs took',end_time-start_time,'seconds')
print("Building the TfIdf Model with vocabulary size = 83700")
start_time = time.time()
idfModel,matrix,mat_norm,hashingTF = build_tf_idf_model(data)
end_time = time.time()
print('Building the tfidf model took',end_time-start_time,'seconds')
while True:
    query = input("Write your query. (If you want to exit write exit)\n")
    if query in ['exit']:
        break
    else:
        return_query(idfModel,hashingTF,query,matrix,mat_norm,prinatable_data)
        input_2 = input("Write 1 to do another query and 0 to exit")
        if int(input_2)==0:
            break
        else:
            os.system('clear')
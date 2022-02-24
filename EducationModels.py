import pandas as pd
import numpy as np
import re
import nltk
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
nltk.download('wordnet')
nltk.download('omw-1.4')

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

#Defining Sentiment Function
def sentiment(x):
    polarity_value = TextBlob(x).sentiment.polarity
    if polarity_value >=0.05:
        return "Positive"
    elif polarity_value <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def sentimentAnalyzer(Data_txt):
    Data_txt['ugc'].isna().sum()
    Data_txt = Data_txt[Data_txt['ugc'].notnull()]
    Data_txt['aspectid']="386,387,388,389,390"

    cleaned1 = lambda x: text_clean_1(x)

    #Lemmatization, Stemming
    stop = stopwords.words('english')
    Data_txt['ugc'] = Data_txt['ugc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    wl = WordNetLemmatizer()
    Data_txt['ugc'] = Data_txt['ugc'].apply(lambda x: " ".join([wl.lemmatize(word) for word in x.split()]))
    Data_txt['ugc'] = Data_txt['ugc'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()
    #display(Data_txt['ugc'])
    Data_txt["ugc"]= Data_txt["ugc"].astype(str)

    #Predicting for Academic Sentiment
    Data_txt['Academic Sentiment'] = 0
    cv = cPickle.load(open('content/2022_02_24/TF_Vectorizer_Academic.pickle','rb'))
    for i in range(len(Data_txt)):
        nltk_tokens = nltk.sent_tokenize(Data_txt.iloc[i,0])
        for j in nltk_tokens:
            sentence = cv.transform(list([j])).toarray()
            price_classifier = cPickle.load(open('content/2022_02_24/RF_Academic_Sentiment_Model.pickle','rb'))
            Yprice = price_classifier.predict(sentence)
            for k in range(len(sentence)):
                Data_txt.iloc[i,2] = Yprice[k]

    Data_txt['Academic Sentiment'].value_counts()
    
    #Predicting for Campus Sentiment
    Data_txt['Campus Sentiment'] = 0
    Data_txt['ugc'] = Data_txt['ugc'].astype(str)
    t_service = []
    cv = cPickle.load(open('content/2022_02_24/TF_Vectorizer_Campus.pickle','rb'))
    for i in range(len(Data_txt)):
        nltk_tokens = nltk.sent_tokenize(Data_txt.iloc[i,0])
        for j in nltk_tokens:
            sentence = cv.transform(list([j])).toarray()
            service_classifier = cPickle.load(open('content/2022_02_24/RF_Campus_Sentiment_Model.pickle','rb'))
            Yservice = service_classifier.predict(sentence)
            for k in range(len(sentence)):
                Data_txt.iloc[i,3] = Yservice[k]
                t_service.append(Yservice[k])

    Data_txt['Campus Sentiment'].value_counts()

    #Predicting for Fees Sentiment
    Data_txt['Fees Sentiment'] = 0
    Data_txt['ugc'] = Data_txt['ugc'].astype(str)
    cv = cPickle.load(open('content/2022_02_24/TF_Vectorizer_Fee.pickle','rb'))
    for i in range(len(Data_txt)):
        nltk_tokens = nltk.sent_tokenize(Data_txt.iloc[i,0])
        for j in nltk_tokens:
            sentence = cv.transform(list([j])).toarray()
            quality_classifier = cPickle.load(open('content/2022_02_24/RF_Fee_Sentiment_Model.pickle','rb'))
            Yquality = quality_classifier.predict(sentence)
            for k in range(len(sentence)):
                Data_txt.iloc[i,4] = Yquality[k]

    Data_txt['Fees Sentiment'].value_counts()

    #Predicting for Online Learning Sentiment
    Data_txt['Online Learning Sentiment'] = 0
    Data_txt['ugc'] = Data_txt['ugc'].astype(str)
    t_delivery = []
    cv = cPickle.load(open('content/2022_02_24/TF_Vectorizer_Online_learning.pickle','rb'))
    for i in range(len(Data_txt)):
        nltk_tokens = nltk.sent_tokenize(Data_txt.iloc[i,0])
        for j in nltk_tokens:
            sentence = cv.transform(list([j])).toarray()
            delivery_classifier = cPickle.load(open('content/2022_02_24/RF_Online_learning_Sentiment_Model.pickle','rb'))
            Ydelivery = delivery_classifier.predict(sentence)
            for k in range(len(sentence)):
                Data_txt.iloc[i,5] = Ydelivery[k]
                t_delivery.append(Ydelivery[k])

    Data_txt['Online Learning Sentiment'].value_counts()

    #Predicting for Placement Sentiment
    Data_txt['Placement Sentiment'] = 0
    Data_txt['ugc'] = Data_txt['ugc'].astype(str)
    t_features = []
    cv = cPickle.load(open('content/2022_02_24/TF_Vectorizer_Placements.pickle','rb'))
    for i in range(len(Data_txt)):
        nltk_tokens = nltk.sent_tokenize(Data_txt.iloc[i,0])
        for j in nltk_tokens:
            sentence = cv.transform(list([j])).toarray()
            features_classifier = cPickle.load(open('content/2022_02_24/RF_Placements_Sentiment_Model.pickle','rb'))
            Yfeatures = features_classifier.predict(sentence)
            for k in range(len(sentence)):
                Data_txt.iloc[i,6] = Yfeatures[k]
                t_features.append(Yfeatures[k])

    Data_txt['Placement Sentiment'].value_counts()

    # To find over_all sentiment
    Data_txt['overallsentiment'] = Data_txt['ugc'].apply(sentiment)
    Data_txt['overallsentiment'].value_counts()
    Data_Final = Data_txt.copy(deep=True) 
    Data_copy_1 = Data_Final.copy(deep=True)
    Data_copy_1 = Data_copy_1.melt(id_vars=["ugc",'aspectid','overallsentiment'],
                    var_name = "identifiedaspectid",
                    value_name = 'aspectsentiment')
    Data_copy_1['identifiedaspectid'] = Data_copy_1['identifiedaspectid'].replace("Academic Sentiment",386)
    Data_copy_1['identifiedaspectid'] = Data_copy_1['identifiedaspectid'].replace("Campus Sentiment",387)
    Data_copy_1['identifiedaspectid'] = Data_copy_1['identifiedaspectid'].replace("Fees Sentiment",388)
    Data_copy_1['identifiedaspectid'] = Data_copy_1['identifiedaspectid'].replace("Online Learning Sentiment",389)
    Data_copy_1['identifiedaspectid'] = Data_copy_1['identifiedaspectid'].replace("Placement Sentiment",390)
    Data_copy_1.drop(columns=['aspectid'],inplace=True)
    Data_copy_2=Data_copy_1[Data_copy_1['aspectsentiment']!='Not Mentioned']
    return Data_copy_2

##if __name__=="__main__":
##    Data_txt={"ugc":["Awesome Campus"]}
##    Data_txt=pd.DataFrame(Data_txt)
##    Data_txt.set_index("ugc")
##    predictedSentiment=sentimentAnalyzer(Data_txt)

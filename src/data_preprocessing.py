import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df_true, df_fake):
    df_fake['target'] = 0
    df_true['target'] = 1
    df = pd.concat([df_true, df_fake]).reset_index(drop=True)
    df['content'] = df['title'] + ' ' + df['text']
    df['clean_text'] = df['content'].apply(preprocess_text)
    return df

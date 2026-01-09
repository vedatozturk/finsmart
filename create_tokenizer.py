import pandas as pd
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('bitcoin_sentiments_21_24.csv')
df.columns = ['date', 'text', 'sentiment_score']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text

df['clean_text'] = df['text'].apply(clean_text)
texts = df['clean_text'].values

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(" tokenizer.pickle oluşturuldu ")
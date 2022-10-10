#!/usr/bin/env python
# coding: utf-8

# ## 0. Data Pre-processing

# In[1]:


import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import re
import spacy
import nltk
nltk.download('all')
nltk.download('stopwords')  
nltk.download('wordnet')


# In[2]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


# In[3]:


data = pd.read_csv("/Users/sapry/Downloads/archive (3)/reddit_wsb.csv")
data = pd.DataFrame(data)
data.head()


# In[4]:


data['original_body'] = data['body']
data.dropna(subset=['body'], inplace=True)
data.head()


# In[5]:


nlp = spacy.blank('en')


# In[6]:


def url(text):
    regex = re.compile(r'https?://\S+|www\.\S+')
    return regex.sub(r'', text)


# In[7]:


def punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# In[8]:


def stop_words(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])


# In[9]:


remove_spaces = lambda x : re.sub('\\s+', ' ', x)


# In[10]:


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                              u"\u23cf"
                              u"\u23e9"
                              u"\u231a"
                              u"\ufe0f"  # dingbats
                              u"\u3030"
                              "]+", flags=re.UNICODE)                      
    return emoji_pattern.sub(r'', string)


# In[11]:


remove_double_quotes = lambda x : x.replace('"', '')
remove_single_quotes = lambda x : x.replace('\'', '')
trim = lambda x : x.strip()

other_chars = ['*', '#', '&x200B', '[', ']', '; ',' ;' "&nbsp", "“","“","”", "x200b"]
def remove_other_chars(x: str):
    for char in other_chars:
        x = x.replace(char, '')
    
    return x

def lower_case_text(text):
    return text.lower()


# In[12]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()


# In[13]:


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # stemming and lemmatization
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# In[14]:


funcs = [
    lower_case_text,
    remove_other_chars,
    clean,
    remove_spaces,
    trim,
    url, 
    punctuation,
    stop_words, 
    remove_emoji, 
    remove_double_quotes, 
    remove_single_quotes,
    ]

for fun in funcs:
    data['body'] = data['body'].apply(fun)


# In[15]:


for fun in funcs:
    data['title'] = data['title'].apply(fun)


# In[16]:


data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

data


# In[17]:


''.join(char for char in data.body.loc[0] if char in string.printable)


# In[18]:


''.join(char for char in data.title.loc[0] if char in string.printable)


# In[19]:


body_list = data.body.tolist()
title_list = data.title.tolist()


# In[20]:


body_list[0]


# In[21]:


title_list[0]


# ## 1. Word Cloud

# In[27]:


conda install -c conda-forge wordcloud


# In[28]:


from wordcloud import WordCloud


# #### Word Cloud for the body

# In[29]:


body_clean = [clean(doc).split() for doc in body_list] 


# In[30]:


long_string = ",".join([" ".join(sentence) for sentence in body_clean])
#print('long_string: \n\n', long_string)

wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)

wordcloud.to_image()


# #### Word Cloud for the title

# In[31]:


title_clean = [clean(doc).split() for doc in title_list] 
long_string = ",".join([" ".join(sentence) for sentence in title_clean])
#print('long_string: \n\n', long_string)


# In[32]:


wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)

wordcloud.to_image()


# #### Positive/Negative statistics

# In[22]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as sentiment_analyzer


# In[23]:


body_data = pd.DataFrame(body_list)
body_data.columns = ['body']
body_data.head()


# In[38]:


sentiment = sentiment_analyzer()

body_data['sentiments'] = body_data['body'].apply(lambda x: sentiment.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
body_data.head()


# In[39]:


body_data['Positive Sentiment'] = body_data['sentiments'].apply(lambda x: x['pos']+1*(10**-4)) 
body_data['Neutral Sentiment'] = body_data['sentiments'].apply(lambda x: x['neu']+1*(10**-4))
body_data['Negative Sentiment']   = body_data['sentiments'].apply(lambda x: x['neg']+1*(10**-4))
body_data.drop(columns=['sentiments'],inplace=True)

#create positivity feature by taking the maximum value among positive, neutral and negative features 
body_data['positivity']=body_data[['Positive Sentiment','Neutral Sentiment','Negative Sentiment']].idxmax(axis=1)
body_data = body_data[body_data['positivity'] != 'Neutral Sentiment']


# In[40]:


body_data.head()


# In[ ]:


#Converting positivity into 1(positive) and 0 (negative)
#body_data['positivity'][body_data['positivity'] == "Positive Sentiment"] = 1
#body_data['positivity'][body_data['positivity'] == "Negative Sentiment"] = 0


# In[30]:


import matplotlib.pyplot as plt


# Think of plotting with neutral sentiment

# In[41]:


body_data.positivity.value_counts().plot.bar()


# In[ ]:





# In[ ]:





# In[ ]:





# ### 2. Topic Moelling with LDA

# In[63]:


conda install -c conda-forge gensim


# In[53]:


import gensim
from gensim import corpora


# In[54]:


dictionary = corpora.Dictionary(body_clean)


# In[55]:


dictionary.doc2bow(body_clean[0])

body_term_matrix = [dictionary.doc2bow(doc) for doc in body_clean]
print(body_term_matrix)

[[(dictionary[id], freq) for id, freq in cp] for cp in body_term_matrix[0:]]


# In[63]:


Lda = gensim.models.ldamodel.LdaModel

lda_model = Lda(body_term_matrix, num_topics=5, id2word = dictionary, passes=10)


# In[71]:


#print(lda_model.print_topics(num_topics=20, num_words=5))


# In[ ]:


#print('\nPerplexity: ', lda_model.log_perplexity(body_term_matrix))  


# In[64]:


print(lda_model.print_topics(num_topics=5, num_words=5))


# #### Display top-10 most significant words for each of the 5 topics

# In[66]:


for (topic, words) in lda_model.print_topics():
    print(topic+1, ":", words, '\n\n')


# In[109]:


from gensim.models import CoherenceModel


# In[ ]:


coherence_values = []
model_list = []

for num_topics in range(2, 25, 3):
    lda_model = Lda(body_term_matrix, num_topics=num_topics, id2word = dictionary, passes=10)
    model_list.append(lda_model)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=body_term_matrix, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_values.append(coherence_lda)


# In[105]:


model_list, coherence_values = coherence_values(texts=body_clean, limit=20)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies_credits = movies.merge(credits, on='title')


# In[4]:


movies_credits = movies_credits[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[5]:


movies_credits.isnull().sum()


# In[6]:


movies_credits.dropna(inplace=True)


# In[7]:


movies_credits.duplicated().sum()


# In[8]:


movies.iloc[0].genres


# In[9]:


import ast
def convert(obj):
    res = []
    for i in ast.literal_eval(obj):
        res.append(i['name'])
    return res


# In[10]:


movies_credits['genres'] = movies_credits['genres'].apply(convert)


# In[11]:


movies_credits['keywords'] = movies_credits['keywords'].apply(convert)


# In[12]:


import ast
def convert3(obj):
    res = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            res.append(i['name'])
            count += 1
        else:
            break
    return res


# In[13]:


movies_credits['cast'] = movies_credits['cast'].apply(convert3)


# In[14]:


import ast
def fetchDirector(obj):
    res = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            res.append(i['name'])
            break
    return res


# In[15]:


movies_credits['crew'] = movies_credits['crew'].apply(fetchDirector)


# In[16]:


movies_credits['overview'] = movies_credits['overview'].apply(lambda x: x.split())


# In[17]:


movies_credits['genres'] = movies_credits['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_credits['keywords'] = movies_credits['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_credits['cast'] = movies_credits['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_credits['crew'] = movies_credits['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[18]:


movies_credits['tags'] = movies_credits['overview'] + movies_credits['genres'] + movies_credits['keywords'] + movies_credits['cast'] + movies_credits['crew']


# In[19]:


new_df = movies_credits[['movie_id', 'title', 'tags']]


# In[20]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[21]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[22]:


import nltk


# In[23]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[24]:


def stem(text):
    res = []
    for i in text.split():
        res.append(ps.stem(i))
    return " ".join(res)


# In[25]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[27]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


similarity = cosine_similarity(vectors)


# In[30]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[31]:


recommend('Iron Man')


# In[32]:


import pickle


# In[34]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[33]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
credits = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits.head(7)
print("Credits Dataframe:", credits.shape)
print("Movies Dataframe:", movies_df.shape)
credits_column_renamed = credits.rename(columns ={'movie_id' :'id'}, inplace = False)
#rename movie_id in credits table as 'id' and then merge the datasets
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_df_merge = movies_df.merge(credits_column_renamed, on='id')
movies_df_merge.head()
#Since we do not require information from all the columns
movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned_df.head()
movies_cleaned_df.head(1)['overview']


from sklearn.feature_extraction.text import TfidfVectorizer
   tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',  #Removes unncesessary symbols
                        ngram_range=(1, 3),      #taking combination of 1-3 different words
                        stop_words = 'english')  #Removes unnecessary charecters

    movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')  # Filling NaNs with empty string
    tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])   # Fitting the TF-IDF on the 'overview' text
    tfv_matrix.shape


# In[2]:


from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
indices


# In[10]:


def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))   #Provides index for each value

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]

# Testing our content-based recommendation system with the seminal film A Beautifuul Mind
give_rec('A Beautiful Mind')


# In[ ]:





# predicting-political-affiliation-twitter
In this project we will be implementing a Deep Learning Neural Network Algorithm to predict Twitter users’ political affiliations using a singular tweet.

To read an in depth writeup on this project, check out: 
https://docs.google.com/document/d/1O7AxPipTLktGYl2PvFXN18ok87yBWgHjhyUGj_Jld44/edit

In order to run this model, you must follow these steps:
1. Install the packages in requirements.txt
2. Download all_tweet_ids.csv and full_member_info.csv from TwitterPoliticians.org
3. Using twarc, hydrate the tweets in all_tweet_ids.csv (warning, this steps creates a file containing 11 
million tweets that is over 20 GB and may take up to 48 hours)
4. Use generate_data.py and generate_tweets.py to filter for US politicians and generate training and testing
data
5. Download glove.6B.300d.txt, and use generate_vocab.py to generate the vocabulary and GLoVe embeddings for the dataset
6. Run model.py to train and test the model :)

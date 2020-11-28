import pandas as pd
from twarc import Twarc
import glob
import json
import os.path

#clean member info
member_info = pd.read_csv('data/full_member_info.csv', encoding='utf16', engine='python')
member_info_filtered = member_info[['country', 'name', 'party', 'uid']]
has_uid = member_info_filtered['uid'] == member_info_filtered['uid']
member_info_filtered = member_info_filtered[has_uid]
is_american = member_info_filtered['country'] == 'United States'
member_info_american = member_info_filtered[is_american]
member_info_american.uid = member_info_american.uid.astype(int)

tweet_texts = pd.DataFrame(columns=['name', 'party', 'uid', 'full_text'])
with open('data/all_tweets.jsonl') as f:
    for line in f:
        tweet = json.loads(line)
        tweet_dict = {}
        tweet_dict['uid'] = tweet['user']['id']
        tweet_dict['full_text'] = tweet['full_text']
        tweet = pd.DataFrame(tweet_dict, index=[0])
        tweet.uid = tweet.uid.astype(int)
        uid = int(tweet['uid'])
        member_info = member_info_american.loc[member_info_american['uid'] == uid]
        if not member_info.empty:
            member_info = member_info.head(1)
            name = member_info.iloc[0]['name']
            party = member_info.iloc[0]['party']
            uid =member_info.iloc[0]['uid']
            full_text = tweet.iloc[0]['full_text']
            tweet_texts = tweet_texts.append({'name': name, 'party': party, 'uid': uid, 'full_text': full_text}, ignore_index=True)

tweet_texts.to_pickle("data/all_tweets.pkl")

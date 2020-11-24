import pandas as pd
#clean tweets
tweets = pd.read_json('data/tweets.jsonl', orient='columns', lines=True)
users = pd.json_normalize(tweets['user'])
tweets['uid'] = users['id']
tweets = tweets[['uid', 'full_text']]

#clean member info
member_info = pd.read_csv('data/full_member_info.csv', encoding='utf16', engine='python')
member_info_filtered = member_info[['country', 'name', 'party', 'uid']]
#filter for uid
has_uid = member_info_filtered['uid'] == member_info_filtered['uid']
member_info_filtered = member_info_filtered[has_uid]
# filter for american
is_american = member_info_filtered['country'] == 'United States'
member_info_american = member_info_filtered[is_american]
# match types
member_info_american.uid = member_info_american.uid.astype(int)

# Join the member information with the tweets 
df = pd.merge(member_info_american, tweets, how='left', on='uid')
df.to_pickle("data/tweets.pkl")
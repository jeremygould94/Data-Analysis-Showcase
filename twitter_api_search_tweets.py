# ----- Twitter API Test - Search Tweets by Keyword -----

# Import stuff
import tweepy
import configparser
import pandas as pd

# Read credentials from config
config = configparser.ConfigParser()
config.read("config.ini")

api_key = config["twitter"]["api_key"]
api_key_secret = config["twitter"]["api_key_secret"]

access_token = config["twitter"]["access_token"]
access_token_secret = config["twitter"]["access_token_secret"]

# Authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Search tweets - cursor is used to increase available tweets from 100 limit - limit is the most recent tweets
keywords = "@gcntweet"
limit = 1000
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode="extended").items(limit)

# Create dataframe to store tweet data and output to csv
columns = ["Tweet_ID", "Tweeted_at", "User", "Tweet"]
data = []
for tweet in tweets:
    data.append([tweet.id_str, tweet.created_at, tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

df.to_csv("keyword_search_tweets.csv", index=False)

# ----- Twitter API Test - Tweets by User -----

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

# User tweets - cursor is used to increase available tweets from 200 limit - limit is the most recent tweets
user = "BBCWorld"
limit = 10
tweets = tweepy.Cursor(api.user_timeline, screen_name=user, count=200, tweet_mode="extended").items(limit)

# If we need less than 200, we can just use this
# tweets = api.user_timeline(screen_name=user, count=limit, tweet_mode="extended")

# Create dataframe to store tweet data and output to csv
columns = ["Tweet_ID", "Tweeted_at", "User", "Tweet"]
data = []
for tweet in tweets:
    data.append([tweet.id_str, tweet.created_at, tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

df.to_csv("tweets_by_user.csv", index=False)

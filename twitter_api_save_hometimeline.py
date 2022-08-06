# ----- Twitter API Test to save Home Timeline Tweets -----

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
public_tweets = api.home_timeline()

# # Display tweets
# # Selected tweet
# print(public_tweets[0].text)
# # All public tweets
# for tweet in public_tweets:
#     print(tweet.text)

# Create dataframe to store tweet data and output to csv
columns = ["Tweet_ID", "Tweeted_at", "User", "Tweet"]
data = []
for tweet in public_tweets:
    data.append([tweet.id_str, tweet.created_at, tweet.user.screen_name, tweet.text])

df = pd.DataFrame(data, columns=columns)

df.to_csv("home_timeline_tweets.csv", index=False)

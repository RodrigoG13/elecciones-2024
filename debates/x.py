import tweepy
import datetime
import json

# Autenticación con la API de Twitter (v2)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAlYuAEAAAAAk9Mrf1337apPSYsAcEu0RFsSw60%3D4ZZfFU6P9IlVi4pGAUCoQw02REsUAx9LdHlWeyIlUYxABJipmd'

client = tweepy.Client(bearer_token=bearer_token)

# Fecha de búsqueda
search_date = datetime.datetime(2024, 4, 7)
search_date_str = search_date.strftime('%Y-%m-%d')

# Realizar la búsqueda
query = '#DebateINE'
start_time = f"{search_date_str}T00:00:00Z"
end_time = f"{(search_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00Z"

tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=['created_at', 'text'])

# Lista para almacenar los tweets
tweets_data = []

# Recopilar información de los tweets
for tweet in tweets.data:
    tweet_info = {
        'created_at': tweet.created_at,
        'text': tweet.text,
        'author_id': tweet.author_id,
    }
    tweets_data.append(tweet_info)

# Guardar los datos en un archivo JSON
with open('tweets.json', 'w', encoding='utf-8') as file:
    json.dump(tweets_data, file, ensure_ascii=False, indent=4)

print("Tweets guardados en tweets.json")

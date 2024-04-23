import os
import random
import asyncio
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv
import praw

@dataclass
class RedditPost:
    post_id: str
    title: str
    url: str
    poster_name: str
    post_text: str = ''

    def __repr__(self):
        return f"Title: {self.title}, URL: {self.url}, Poster Name: {self.poster_name}"

class RedditPostFetcher:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = 'windows:reddit_access:v1.0 (by u/Proud-Election-4152)'
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False
        )

    def fetch_text_posts_from_front_page(self, limit=10) -> List[RedditPost]:
        results: List[RedditPost] = []

        for post in self.reddit.front.hot(limit=limit):
            if post.is_self:
                results.append(RedditPost(
                    post_id=post.id,
                    title=post.title.encode("utf-8", "ignore").decode("utf-8"),
                    url=post.url,
                    poster_name=post.author.name,
                    post_text=post.selftext,
                ))

        return results

    def get_random_reddit_post(self, limit=100) -> RedditPost:
        posts = []
        
        for post in self.reddit.front.hot(limit=limit):
            if post.is_self:
                posts.append(RedditPost(
                    post_id=post.id,
                    title=post.title.encode("utf-8", "ignore").decode("utf-8"),
                    url=post.url,
                    poster_name=post.author.name,
                    post_text=post.selftext
                ))
        
        return random.choice(posts) if posts else None

if __name__ == "__main__":
    reddit_fetcher = RedditPostFetcher()
    result = reddit_fetcher.get_random_reddit_post()

    if result:
        print(result)
    else:
        print("No self-posts found.")

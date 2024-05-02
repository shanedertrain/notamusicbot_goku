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
    def __init__(self, post_limit=300):
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
        self.post_generator = self.posts_traverser()

    def fetch_text_posts_from_front_page(self, limit=100) -> List[RedditPost]:
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

    def posts_traverser(self):
        for post in random.shuffle(self.fetch_text_posts_from_front_page()):
            yield post

    def get_random_reddit_post(self) -> RedditPost:
        return random.choice(self.posts) if self.posts else None

if __name__ == "__main__":
    reddit_fetcher = RedditPostFetcher()

    # Example of using the generator with enumerate
    for idx, post in enumerate(reddit_fetcher.posts_traverser()):
        print(f"Post {idx + 1}: {post}")

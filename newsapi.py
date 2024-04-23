import os
from enum import Enum, auto
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import random
from typing import List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Article:
    title: str
    source: str
    description: str
    url: str
    text: str = None

class Category(Enum):
    BUSINESS = auto()
    ENTERTAINMENT = auto()
    GENERAL = auto()
    HEALTH = auto()
    SCIENCE = auto()
    SPORTS = auto()
    TECHNOLOGY = auto()

class CountryCode(Enum):
    US = 'us'
    UK = 'gb'
    CANADA = 'ca'
    AUSTRALIA = 'au'
    INDIA = 'in'
    GERMANY = 'de'
    FRANCE = 'fr'

class NewsScraper:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')

    def get_articles(self, country_code: CountryCode = CountryCode.US, category: Optional[Category] = None) -> List[Article]:
        if isinstance(country_code, CountryCode):
            country_code = country_code.value
        url = f'https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={self.api_key}'
        if category:
            url += f'&category={category.name.lower()}'
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok':
            articles_data = data['articles']
            articles = [Article(article['title'], article['source']['name'], article['description'], article['url']) for article in articles_data]
            return articles
        else:
            print('Failed to fetch news headlines:', data['message'])
            return []

    def get_random_article(self, country_code: CountryCode = CountryCode.US, category: Optional[Category] = None) -> Article:
        articles = self.get_articles(country_code=country_code, category=category)
        return random.choice(articles)
    
    def get_article_text(self, article: Article) -> str:
        response = requests.get(article.url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Assuming the article text is contained within <p> tags
        article_paragraphs = soup.find_all('p')
        article_text = ' '.join([paragraph.get_text() for paragraph in article_paragraphs])
        return article_text
    

if __name__ == "__main__":
    news_scraper = NewsScraper()
    article = news_scraper.get_random_article(country_code=CountryCode.UK, category=Category.TECHNOLOGY)
    article_text = news_scraper.get_article_text(article)
    print("Article Text:", article_text)
    print(article)

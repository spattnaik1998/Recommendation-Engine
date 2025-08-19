#!/usr/bin/env python3
"""
Podcast Service Module
Handles podcast-related operations including search, retrieval, and analytics.
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional
import os
from collections import Counter
import re

logger = logging.getLogger(__name__)

class PodcastService:
    """Service for managing podcast operations."""
    
    def __init__(self):
        self.data_file = "data/podcasts_database.json"
        self.podcasts = []
        self.podcasts_by_id = {}
        # Using Listen Notes API (free tier allows 100 requests/month)
        self.api_base_url = "https://listen-api.listennotes.com/api/v2"
        
    def initialize(self) -> None:
        """Initialize the podcast service by loading podcasts."""
        self.load_podcasts()
        if not self.podcasts:
            logger.info("No podcasts found, scraping from API...")
            self.scrape_podcasts_from_api()
        
    def load_podcasts(self) -> None:
        """Load podcasts from the database file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.podcasts = json.load(f)
                
                # Create ID lookup dictionary
                self.podcasts_by_id = {podcast.get('id', str(i)): podcast for i, podcast in enumerate(self.podcasts)}
                
                logger.info(f"Loaded {len(self.podcasts)} podcasts from database")
            else:
                logger.warning(f"Database file {self.data_file} not found")
                self.podcasts = []
                self.podcasts_by_id = {}
                
        except Exception as e:
            logger.error(f"Error loading podcasts: {e}")
            self.podcasts = []
            self.podcasts_by_id = {}
    
    def save_podcasts(self) -> None:
        """Save podcasts to the database file."""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.podcasts, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.podcasts)} podcasts to database")
        except Exception as e:
            logger.error(f"Error saving podcasts: {e}")
    
    def scrape_podcasts_from_api(self) -> Dict:
        """Scrape podcasts from free APIs."""
        try:
            # Get podcasts from iTunes API (free, no key required)
            podcasts_data = self._scrape_from_itunes()
            
            if not podcasts_data:
                logger.warning("No podcasts could be scraped from iTunes API")
                return {
                    'status': 'error',
                    'count': 0,
                    'message': 'Failed to scrape podcasts from iTunes API'
                }
            
            self.podcasts = podcasts_data
            self.podcasts_by_id = {podcast.get('id', str(i)): podcast for i, podcast in enumerate(self.podcasts)}
            self.save_podcasts()
            
            return {
                'status': 'success',
                'count': len(self.podcasts),
                'message': f'Successfully loaded {len(self.podcasts)} podcasts from iTunes API'
            }
            
        except Exception as e:
            logger.error(f"Error scraping podcasts: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _scrape_from_itunes(self) -> List[Dict]:
        """Scrape podcasts from iTunes API."""
        podcasts = []
        
        # Science, Technology, and AI focused search terms
        search_terms = [
            'artificial intelligence', 'machine learning', 'technology', 'science',
            'computer science', 'programming', 'software engineering', 'data science',
            'robotics', 'automation', 'cybersecurity', 'blockchain', 'quantum computing',
            'physics', 'chemistry', 'biology', 'mathematics', 'engineering',
            'space science', 'astronomy', 'astrophysics', 'nanotechnology',
            'biotechnology', 'renewable energy', 'climate science', 'innovation',
            'tech news', 'scientific research', 'STEM education'
        ]
        
        try:
            for term in search_terms:
                if len(podcasts) >= 1000:  # Limit to 1000 podcasts
                    break
                    
                url = f"https://itunes.apple.com/search"
                params = {
                    'term': term,
                    'media': 'podcast',
                    'limit': 50,
                    'country': 'US'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('results', []):
                        if len(podcasts) >= 1000:
                            break
                            
                        podcast = self._format_itunes_podcast(item, term)
                        if podcast and not any(p['id'] == podcast['id'] for p in podcasts):
                            podcasts.append(podcast)
                
                # Rate limiting
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error scraping from iTunes: {e}")
        
        logger.info(f"Scraped {len(podcasts)} podcasts from iTunes API")
        return podcasts
    
    def _format_itunes_podcast(self, item: Dict, category: str) -> Optional[Dict]:
        """Format iTunes API response into our podcast format."""
        try:
            return {
                'id': str(item.get('collectionId', '')),
                'title': item.get('collectionName', ''),
                'description': item.get('description', ''),
                'author': item.get('artistName', ''),
                'publisher': item.get('artistName', ''),
                'categories': [category.title(), item.get('primaryGenreName', '')],
                'language': item.get('country', 'US').lower(),
                'image_url': item.get('artworkUrl600', item.get('artworkUrl100', '')),
                'feed_url': item.get('feedUrl', ''),
                'website_url': item.get('collectionViewUrl', ''),
                'episode_count': item.get('trackCount', 0),
                'rating': 0,  # iTunes doesn't provide ratings in search
                'rating_count': 0,
                'explicit': item.get('collectionExplicitness', 'notExplicit') == 'explicit',
                'last_updated': item.get('releaseDate', ''),
                'popularity_score': 0
            }
        except Exception as e:
            logger.error(f"Error formatting iTunes podcast: {e}")
            return None
    
    
    def get_podcast_count(self) -> int:
        """Get the total number of podcasts."""
        return len(self.podcasts)
    
    def get_all_podcasts(self) -> List[Dict]:
        """Get all podcasts."""
        return self.podcasts
    
    def get_podcast_by_id(self, podcast_id: str) -> Optional[Dict]:
        """Get a podcast by its ID."""
        return self.podcasts_by_id.get(podcast_id)
    
    def search_podcasts(self, query: str, limit: int = 10) -> List[Dict]:
        """Search podcasts by title, author, or category."""
        if not query.strip():
            return []
        
        query_lower = query.lower().strip()
        results = []
        
        for podcast in self.podcasts:
            score = 0
            
            # Search in title
            title = podcast.get('title', '').lower()
            if query_lower in title:
                score += 10
                if title.startswith(query_lower):
                    score += 5
            
            # Search in author
            author = podcast.get('author', '').lower()
            if query_lower in author:
                score += 8
                if author.startswith(query_lower):
                    score += 3
            
            # Search in categories
            categories = [cat.lower() for cat in podcast.get('categories', [])]
            for category in categories:
                if query_lower in category:
                    score += 5
            
            # Search in description
            description = podcast.get('description', '').lower()
            if query_lower in description:
                score += 2
            
            # Search in publisher
            publisher = podcast.get('publisher', '').lower()
            if query_lower in publisher:
                score += 1
            
            if score > 0:
                podcast_result = podcast.copy()
                podcast_result['search_score'] = score
                results.append(podcast_result)
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results[:limit]
    
    def get_podcasts_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """Get podcasts by category."""
        category_lower = category.lower()
        results = []
        
        for podcast in self.podcasts:
            categories = [cat.lower() for cat in podcast.get('categories', [])]
            if any(category_lower in cat for cat in categories):
                results.append(podcast)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_popular_podcasts(self, limit: int = 20) -> List[Dict]:
        """Get popular podcasts based on ratings and popularity score."""
        # Filter podcasts with ratings and sort by popularity
        rated_podcasts = [
            podcast for podcast in self.podcasts 
            if podcast.get('rating', 0) > 0 and podcast.get('rating_count', 0) > 0
        ]
        
        # Sort by a combination of rating and popularity
        def popularity_score(podcast):
            rating = podcast.get('rating', 0)
            count = podcast.get('rating_count', 0)
            popularity = podcast.get('popularity_score', 0)
            # Weighted score: rating * log(count + 1) + popularity
            import math
            return rating * math.log(count + 1) + popularity
        
        rated_podcasts.sort(key=popularity_score, reverse=True)
        return rated_podcasts[:limit]
    
    def get_analytics(self) -> Dict:
        """Get analytics data about the podcast collection."""
        if not self.podcasts:
            return {
                'total_podcasts': 0,
                'total_authors': 0,
                'categories': {},
                'languages': {},
                'publishers': {},
                'average_rating': 0,
                'podcasts_with_ratings': 0
            }
        
        # Count unique authors
        all_authors = set()
        for podcast in self.podcasts:
            author = podcast.get('author', '')
            if author:
                all_authors.add(author)
        
        # Count categories
        category_counter = Counter()
        for podcast in self.podcasts:
            for category in podcast.get('categories', []):
                category_counter[category] += 1
        
        # Count languages
        language_counter = Counter()
        for podcast in self.podcasts:
            lang = podcast.get('language', 'unknown')
            language_counter[lang] += 1
        
        # Count publishers
        publisher_counter = Counter()
        for podcast in self.podcasts:
            publisher = podcast.get('publisher', 'Unknown')
            if publisher and publisher != 'Unknown':
                publisher_counter[publisher] += 1
        
        # Calculate average rating
        rated_podcasts = [
            podcast for podcast in self.podcasts 
            if podcast.get('rating', 0) > 0
        ]
        
        avg_rating = 0
        if rated_podcasts:
            total_rating = sum(podcast.get('rating', 0) for podcast in rated_podcasts)
            avg_rating = total_rating / len(rated_podcasts)
        
        return {
            'total_podcasts': len(self.podcasts),
            'total_authors': len(all_authors),
            'categories': dict(category_counter.most_common(20)),
            'languages': dict(language_counter.most_common(10)),
            'publishers': dict(publisher_counter.most_common(10)),
            'average_rating': round(avg_rating, 2),
            'podcasts_with_ratings': len(rated_podcasts),
            'podcasts_with_descriptions': len([p for p in self.podcasts if p.get('description')]),
            'explicit_podcasts': len([p for p in self.podcasts if p.get('explicit', False)])
        }
    
    def refresh_data(self) -> bool:
        """Refresh podcast data by reloading from file."""
        try:
            self.load_podcasts()
            return True
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return False

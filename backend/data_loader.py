#!/usr/bin/env python3
"""
Data Loader Module
Loads book data from external APIs (Google Books API) to populate the RAG database.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading book data from external APIs."""
    
    def __init__(self):
        self.google_books_api_url = "https://www.googleapis.com/books/v1/volumes"
        self.data_file = "data/books_database.json"
        self.batch_size = 40  # Google Books API max results per request
        self.delay_between_requests = 1  # Seconds to avoid rate limiting
        
    def load_books_from_api(self, target_count: int = 1000) -> Dict:
        """Load books from Google Books API."""
        logger.info(f"Starting to load {target_count} books from Google Books API")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Load existing books if any
        existing_books = self._load_existing_books()
        logger.info(f"Found {len(existing_books)} existing books")
        
        if len(existing_books) >= target_count:
            return {
                'status': 'success',
                'message': f'Already have {len(existing_books)} books in database',
                'count': len(existing_books)
            }
        
        books_needed = target_count - len(existing_books)
        logger.info(f"Need to load {books_needed} more books")
        
        # Search queries focused on Science, Technology, and AI
        search_queries = [
            "artificial intelligence", "machine learning", "deep learning", "neural networks",
            "computer science", "programming", "software engineering", "data science",
            "robotics", "automation", "algorithms", "computer vision",
            "natural language processing", "big data", "cloud computing", "cybersecurity",
            "quantum computing", "blockchain", "internet of things", "IoT",
            "physics", "chemistry", "biology", "mathematics", "statistics",
            "engineering", "electrical engineering", "mechanical engineering", "bioengineering",
            "nanotechnology", "biotechnology", "genetics", "neuroscience",
            "space science", "astronomy", "astrophysics", "climate science",
            "renewable energy", "sustainable technology", "green technology",
            "scientific method", "research methodology", "data analysis"
        ]
        
        new_books = []
        books_per_query = max(1, books_needed // len(search_queries))
        
        try:
            for query in search_queries:
                if len(new_books) >= books_needed:
                    break
                    
                logger.info(f"Searching for books with query: '{query}'")
                query_books = self._fetch_books_by_query(
                    query, 
                    max_results=min(books_per_query, books_needed - len(new_books))
                )
                
                new_books.extend(query_books)
                logger.info(f"Total books collected: {len(existing_books) + len(new_books)}")
                
                # Rate limiting
                time.sleep(self.delay_between_requests)
            
            # Combine with existing books
            all_books = existing_books + new_books
            
            # Remove duplicates based on title and author
            unique_books = self._remove_duplicates(all_books)
            
            # Save to file
            self._save_books(unique_books)
            
            logger.info(f"Successfully loaded {len(unique_books)} total books")
            
            return {
                'status': 'success',
                'message': f'Successfully loaded {len(new_books)} new books',
                'total_count': len(unique_books),
                'new_count': len(new_books),
                'existing_count': len(existing_books)
            }
            
        except Exception as e:
            logger.error(f"Error loading books from API: {e}")
            return {
                'status': 'error',
                'message': f'Failed to load books: {str(e)}',
                'count': len(existing_books)
            }
    
    def _fetch_books_by_query(self, query: str, max_results: int = 40) -> List[Dict]:
        """Fetch books from Google Books API for a specific query."""
        books = []
        start_index = 0
        
        while len(books) < max_results:
            try:
                # Calculate how many results to request
                results_needed = min(self.batch_size, max_results - len(books))
                
                params = {
                    'q': query,
                    'startIndex': start_index,
                    'maxResults': results_needed,
                    'printType': 'books',
                    'orderBy': 'relevance'
                }
                
                response = requests.get(self.google_books_api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if 'items' not in data:
                    logger.warning(f"No items found for query: {query}")
                    break
                
                for item in data['items']:
                    book = self._parse_google_book(item)
                    if book and self._is_science_tech_ai_book(book):
                        books.append(book)
                
                # Check if we've reached the end of results
                if len(data['items']) < results_needed:
                    break
                
                start_index += results_needed
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for query '{query}': {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error for query '{query}': {e}")
                break
        
        return books[:max_results]
    
    def _parse_google_book(self, item: Dict) -> Optional[Dict]:
        """Parse a Google Books API item into our book format."""
        try:
            volume_info = item.get('volumeInfo', {})
            
            # Skip books without essential information
            title = volume_info.get('title', '').strip()
            if not title:
                return None
            
            authors = volume_info.get('authors', [])
            if not authors:
                authors = ['Unknown Author']
            
            # Extract other information
            description = volume_info.get('description', '')
            if len(description) > 1000:  # Limit description length
                description = description[:1000] + "..."
            
            categories = volume_info.get('categories', [])
            subjects = []
            for category in categories:
                # Split categories that contain multiple subjects
                if '/' in category:
                    subjects.extend([s.strip() for s in category.split('/')])
                else:
                    subjects.append(category.strip())
            
            # Remove duplicates and limit subjects
            subjects = list(set(subjects))[:10]
            
            published_date = volume_info.get('publishedDate', '')
            
            # Extract year from published date
            pub_year = ''
            if published_date:
                try:
                    if len(published_date) >= 4:
                        pub_year = published_date[:4]
                except:
                    pass
            
            # Get cover image
            image_links = volume_info.get('imageLinks', {})
            cover_url = image_links.get('thumbnail', image_links.get('smallThumbnail', ''))
            
            # Get additional metadata
            page_count = volume_info.get('pageCount', 0)
            language = volume_info.get('language', 'en')
            publisher = volume_info.get('publisher', '')
            
            # Get ratings
            average_rating = volume_info.get('averageRating', 0)
            ratings_count = volume_info.get('ratingsCount', 0)
            
            book = {
                'id': item.get('id', ''),
                'title': title,
                'authors': authors,
                'description': description,
                'subjects': subjects,
                'published_date': published_date,
                'publication_year': pub_year,
                'publisher': publisher,
                'page_count': page_count,
                'language': language,
                'cover_url': cover_url,
                'average_rating': average_rating,
                'ratings_count': ratings_count,
                'source': 'google_books',
                'loaded_at': datetime.now().isoformat()
            }
            
            return book
            
        except Exception as e:
            logger.error(f"Error parsing Google Books item: {e}")
            return None
    
    def _load_existing_books(self) -> List[Dict]:
        """Load existing books from the database file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing books: {e}")
        
        return []
    
    def _save_books(self, books: List[Dict]) -> None:
        """Save books to the database file."""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(books, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(books)} books to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving books: {e}")
            raise
    
    def _remove_duplicates(self, books: List[Dict]) -> List[Dict]:
        """Remove duplicate books based on title and first author."""
        seen = set()
        unique_books = []
        
        for book in books:
            # Create a key based on title and first author
            title = book.get('title', '').lower().strip()
            first_author = book.get('authors', [''])[0].lower().strip()
            key = f"{title}|{first_author}"
            
            if key not in seen:
                seen.add(key)
                unique_books.append(book)
        
        logger.info(f"Removed {len(books) - len(unique_books)} duplicate books")
        return unique_books
    
    def get_book_count(self) -> int:
        """Get the current number of books in the database."""
        books = self._load_existing_books()
        return len(books)
    
    def _is_science_tech_ai_book(self, book: Dict) -> bool:
        """Check if a book is related to Science, Technology, or AI."""
        # Define keywords for Science, Technology, and AI domains
        science_tech_ai_keywords = {
            # AI and Machine Learning
            'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
            'natural language processing', 'computer vision', 'robotics', 'automation',
            'data science', 'big data', 'algorithms', 'ai', 'ml', 'nlp',
            
            # Computer Science and Technology
            'computer science', 'programming', 'software engineering', 'software development',
            'web development', 'mobile development', 'cloud computing', 'cybersecurity',
            'blockchain', 'cryptocurrency', 'quantum computing', 'internet of things',
            'iot', 'databases', 'networking', 'operating systems', 'distributed systems',
            
            # Engineering and Technology
            'engineering', 'electrical engineering', 'mechanical engineering', 'civil engineering',
            'bioengineering', 'chemical engineering', 'aerospace engineering', 'industrial engineering',
            'technology', 'innovation', 'digital transformation', 'tech',
            
            # Sciences
            'physics', 'chemistry', 'biology', 'mathematics', 'statistics', 'calculus',
            'astronomy', 'astrophysics', 'space science', 'earth science', 'climate science',
            'environmental science', 'materials science', 'nanotechnology', 'biotechnology',
            'genetics', 'genomics', 'neuroscience', 'cognitive science', 'bioinformatics',
            
            # Research and Methods
            'scientific method', 'research methodology', 'data analysis', 'statistical analysis',
            'experimental design', 'computational', 'simulation', 'modeling',
            
            # Energy and Sustainability
            'renewable energy', 'sustainable technology', 'green technology', 'solar energy',
            'wind energy', 'nuclear energy', 'energy systems', 'sustainability'
        }
        
        # Get text content to search
        title = book.get('title', '').lower()
        description = book.get('description', '').lower()
        subjects = [s.lower() for s in book.get('subjects', [])]
        authors = [a.lower() for a in book.get('authors', [])]
        
        # Combine all text content
        all_text = f"{title} {description} {' '.join(subjects)} {' '.join(authors)}"
        
        # Check if any science/tech/AI keywords are present
        for keyword in science_tech_ai_keywords:
            if keyword in all_text:
                return True
        
        # Additional check for subjects - look for science/tech categories
        science_tech_subjects = {
            'science', 'technology', 'computer', 'engineering', 'mathematics', 'physics',
            'chemistry', 'biology', 'artificial intelligence', 'machine learning', 'programming',
            'software', 'data', 'research', 'technical', 'scientific', 'computational'
        }
        
        for subject in subjects:
            for sci_subject in science_tech_subjects:
                if sci_subject in subject:
                    return True
        
        # Exclude obvious fiction and non-technical categories
        fiction_keywords = {
            'fiction', 'novel', 'romance', 'mystery', 'thriller', 'fantasy', 'adventure',
            'drama', 'poetry', 'literature', 'story', 'tales', 'narrative', 'memoir',
            'biography', 'autobiography', 'history', 'philosophy', 'religion', 'politics',
            'economics', 'psychology', 'sociology', 'anthropology', 'art', 'music',
            'cooking', 'travel', 'health', 'fitness', 'self-help', 'business management'
        }
        
        # If it contains fiction keywords and no science/tech keywords, exclude it
        fiction_count = sum(1 for keyword in fiction_keywords if keyword in all_text)
        if fiction_count > 0:
            # Only exclude if it has fiction keywords but no strong science/tech indicators
            strong_tech_keywords = {
                'artificial intelligence', 'machine learning', 'computer science', 'programming',
                'software engineering', 'data science', 'robotics', 'algorithms', 'physics',
                'chemistry', 'biology', 'mathematics', 'engineering', 'technology'
            }
            
            has_strong_tech = any(keyword in all_text for keyword in strong_tech_keywords)
            if not has_strong_tech:
                return False
        
        # Default to include if we're not sure (better to be inclusive for science/tech)
        return True

    def get_books(self) -> List[Dict]:
        """Get all books from the database."""
        return self._load_existing_books()

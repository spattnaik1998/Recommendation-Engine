#!/usr/bin/env python3
"""
Book Service Module
Handles book-related operations including search, retrieval, and analytics.
"""

import json
import logging
from typing import Dict, List, Optional
import os
from collections import Counter
import re

logger = logging.getLogger(__name__)

class BookService:
    """Service for managing book operations."""
    
    def __init__(self):
        self.data_file = "data/books_database.json"
        self.books = []
        self.books_by_id = {}
        
    def initialize(self) -> None:
        """Initialize the book service by loading books."""
        self.load_books()
        
    def load_books(self) -> None:
        """Load books from the database file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.books = json.load(f)
                
                # Create ID lookup dictionary
                self.books_by_id = {book.get('id', str(i)): book for i, book in enumerate(self.books)}
                
                logger.info(f"Loaded {len(self.books)} books from database")
            else:
                logger.warning(f"Database file {self.data_file} not found")
                self.books = []
                self.books_by_id = {}
                
        except Exception as e:
            logger.error(f"Error loading books: {e}")
            self.books = []
            self.books_by_id = {}
    
    def get_book_count(self) -> int:
        """Get the total number of books."""
        return len(self.books)
    
    def get_all_books(self) -> List[Dict]:
        """Get all books."""
        return self.books
    
    def get_book_by_id(self, book_id: str) -> Optional[Dict]:
        """Get a book by its ID."""
        return self.books_by_id.get(book_id)
    
    def search_books(self, query: str, limit: int = 10) -> List[Dict]:
        """Search books by title, author, or subject."""
        if not query.strip():
            return []
        
        query_lower = query.lower().strip()
        results = []
        
        for book in self.books:
            score = 0
            
            # Search in title
            title = book.get('title', '').lower()
            if query_lower in title:
                score += 10
                if title.startswith(query_lower):
                    score += 5
            
            # Search in authors
            authors = [author.lower() for author in book.get('authors', [])]
            for author in authors:
                if query_lower in author:
                    score += 8
                    if author.startswith(query_lower):
                        score += 3
            
            # Search in subjects
            subjects = [subject.lower() for subject in book.get('subjects', [])]
            for subject in subjects:
                if query_lower in subject:
                    score += 5
            
            # Search in description
            description = book.get('description', '').lower()
            if query_lower in description:
                score += 2
            
            # Search in publisher
            publisher = book.get('publisher', '').lower()
            if query_lower in publisher:
                score += 1
            
            if score > 0:
                book_result = book.copy()
                book_result['search_score'] = score
                results.append(book_result)
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results[:limit]
    
    def get_books_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """Get books by genre/subject."""
        genre_lower = genre.lower()
        results = []
        
        for book in self.books:
            subjects = [subject.lower() for subject in book.get('subjects', [])]
            if any(genre_lower in subject for subject in subjects):
                results.append(book)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_books_by_author(self, author: str, limit: int = 10) -> List[Dict]:
        """Get books by author."""
        author_lower = author.lower()
        results = []
        
        for book in self.books:
            authors = [a.lower() for a in book.get('authors', [])]
            if any(author_lower in a for a in authors):
                results.append(book)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_popular_books(self, limit: int = 20) -> List[Dict]:
        """Get popular books based on ratings."""
        # Filter books with ratings and sort by average rating and rating count
        rated_books = [
            book for book in self.books 
            if book.get('average_rating', 0) > 0 and book.get('ratings_count', 0) > 0
        ]
        
        # Sort by a combination of rating and popularity
        def popularity_score(book):
            rating = book.get('average_rating', 0)
            count = book.get('ratings_count', 0)
            # Weighted score: rating * log(count + 1)
            import math
            return rating * math.log(count + 1)
        
        rated_books.sort(key=popularity_score, reverse=True)
        return rated_books[:limit]
    
    def get_recent_books(self, limit: int = 20) -> List[Dict]:
        """Get recently published books."""
        # Filter books with publication years and sort by year
        books_with_years = []
        
        for book in self.books:
            pub_year = book.get('publication_year', '')
            if pub_year and pub_year.isdigit():
                year = int(pub_year)
                if year >= 2000:  # Recent books from 2000 onwards
                    book_copy = book.copy()
                    book_copy['year_int'] = year
                    books_with_years.append(book_copy)
        
        books_with_years.sort(key=lambda x: x['year_int'], reverse=True)
        return books_with_years[:limit]
    
    def get_analytics(self) -> Dict:
        """Get analytics data about the book collection."""
        if not self.books:
            return {
                'total_books': 0,
                'total_authors': 0,
                'genres': {},
                'publication_years': {},
                'languages': {},
                'publishers': {},
                'average_rating': 0,
                'books_with_ratings': 0
            }
        
        # Count unique authors
        all_authors = set()
        for book in self.books:
            all_authors.update(book.get('authors', []))
        
        # Count genres/subjects
        genre_counter = Counter()
        for book in self.books:
            for subject in book.get('subjects', []):
                genre_counter[subject] += 1
        
        # Count publication years
        year_counter = Counter()
        for book in self.books:
            year = book.get('publication_year', '')
            if year and year.isdigit():
                decade = (int(year) // 10) * 10
                year_counter[f"{decade}s"] += 1
        
        # Count languages
        language_counter = Counter()
        for book in self.books:
            lang = book.get('language', 'unknown')
            language_counter[lang] += 1
        
        # Count publishers
        publisher_counter = Counter()
        for book in self.books:
            publisher = book.get('publisher', 'Unknown')
            if publisher and publisher != 'Unknown':
                publisher_counter[publisher] += 1
        
        # Calculate average rating
        rated_books = [
            book for book in self.books 
            if book.get('average_rating', 0) > 0
        ]
        
        avg_rating = 0
        if rated_books:
            total_rating = sum(book.get('average_rating', 0) for book in rated_books)
            avg_rating = total_rating / len(rated_books)
        
        return {
            'total_books': len(self.books),
            'total_authors': len(all_authors),
            'genres': dict(genre_counter.most_common(20)),
            'publication_years': dict(year_counter.most_common(10)),
            'languages': dict(language_counter.most_common(10)),
            'publishers': dict(publisher_counter.most_common(10)),
            'average_rating': round(avg_rating, 2),
            'books_with_ratings': len(rated_books),
            'books_with_descriptions': len([b for b in self.books if b.get('description')]),
            'books_with_covers': len([b for b in self.books if b.get('cover_url')])
        }
    
    def get_recommendations_by_book(self, book_id: str, limit: int = 5) -> List[Dict]:
        """Get book recommendations based on a specific book."""
        target_book = self.get_book_by_id(book_id)
        if not target_book:
            return []
        
        target_subjects = set(target_book.get('subjects', []))
        target_authors = set(target_book.get('authors', []))
        
        recommendations = []
        
        for book in self.books:
            if book.get('id') == book_id:
                continue  # Skip the target book itself
            
            score = 0
            
            # Score based on shared subjects
            book_subjects = set(book.get('subjects', []))
            shared_subjects = target_subjects.intersection(book_subjects)
            score += len(shared_subjects) * 3
            
            # Score based on shared authors
            book_authors = set(book.get('authors', []))
            shared_authors = target_authors.intersection(book_authors)
            score += len(shared_authors) * 5
            
            # Bonus for highly rated books
            rating = book.get('average_rating', 0)
            if rating >= 4.0:
                score += 2
            elif rating >= 3.5:
                score += 1
            
            if score > 0:
                book_rec = book.copy()
                book_rec['recommendation_score'] = score
                recommendations.append(book_rec)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return recommendations[:limit]
    
    def get_trending_genres(self, limit: int = 10) -> List[Dict]:
        """Get trending genres based on recent books."""
        # Get books from the last 5 years
        current_year = 2024
        recent_books = []
        
        for book in self.books:
            pub_year = book.get('publication_year', '')
            if pub_year and pub_year.isdigit():
                year = int(pub_year)
                if year >= current_year - 5:
                    recent_books.append(book)
        
        # Count genres in recent books
        genre_counter = Counter()
        for book in recent_books:
            for subject in book.get('subjects', []):
                genre_counter[subject] += 1
        
        # Convert to list of dictionaries
        trending = []
        for genre, count in genre_counter.most_common(limit):
            trending.append({
                'genre': genre,
                'count': count,
                'percentage': round((count / len(recent_books)) * 100, 1) if recent_books else 0
            })
        
        return trending
    
    def refresh_data(self) -> bool:
        """Refresh book data by reloading from file."""
        try:
            self.load_books()
            return True
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return False

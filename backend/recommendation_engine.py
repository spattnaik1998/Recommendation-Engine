#!/usr/bin/env python3
"""
Recommendation Engine Module
RAG-based book recommendation system for the web application.
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import math
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """RAG-based recommendation engine for books."""
    
    def __init__(self):
        self.data_file = "data/books_database.json"
        self.books = []
        self.compressed_books = []  # Store compressed book contexts
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.compressed_tfidf_matrix = None  # TF-IDF matrix for compressed contexts
        self.is_initialized = False
        
    def initialize(self) -> None:
        """Initialize the recommendation engine."""
        try:
            self._load_books()
            if self.books:
                self._prepare_vectors()
                self.is_initialized = True
                logger.info("Recommendation engine initialized successfully")
            else:
                logger.warning("No books loaded - recommendation engine not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            self.is_initialized = False
    
    def _load_books(self) -> None:
        """Load books from the database."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.books = json.load(f)
                logger.info(f"Loaded {len(self.books)} books for recommendations")
            else:
                logger.warning(f"Database file {self.data_file} not found")
                self.books = []
        except Exception as e:
            logger.error(f"Error loading books: {e}")
            self.books = []
    
    def _prepare_vectors(self) -> None:
        """Prepare TF-IDF vectors for content-based recommendations."""
        if not self.books:
            return
        
        # Create compressed book contexts
        self._create_compressed_contexts()
        
        # Combine text features for each book (original approach)
        text_features = []
        compressed_features = []
        
        for i, book in enumerate(self.books):
            # Original features
            features = []
            
            # Add title (with higher weight)
            title = book.get('title', '')
            features.extend([title] * 3)  # Triple weight for title
            
            # Add authors
            authors = book.get('authors', [])
            features.extend(authors)
            
            # Add subjects/genres
            subjects = book.get('subjects', [])
            features.extend(subjects)
            
            # Add description
            description = book.get('description', '')
            if description:
                features.append(description)
            
            # Add publisher
            publisher = book.get('publisher', '')
            if publisher:
                features.append(publisher)
            
            # Combine all features
            combined_text = ' '.join(features)
            text_features.append(combined_text)
            
            # Add compressed features
            compressed_book = self.compressed_books[i]
            compressed_text = compressed_book.get('compressed_context', combined_text)
            compressed_features.append(compressed_text)
        
        # Create TF-IDF matrices for both original and compressed contexts
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Create compressed TF-IDF vectorizer with optimized parameters
        self.compressed_vectorizer = TfidfVectorizer(
            max_features=3000,  # Smaller feature space for compressed content
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better semantic capture
            min_df=1,
            max_df=0.7,
            sublinear_tf=True  # Use sublinear TF scaling
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        self.compressed_tfidf_matrix = self.compressed_vectorizer.fit_transform(compressed_features)
        
        logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
        logger.info(f"Created compressed TF-IDF matrix with shape {self.compressed_tfidf_matrix.shape}")
    
    def get_recommendations(self, preferences: Dict, num_recommendations: int = 3) -> List[Dict]:
        """Get book recommendations based on user preferences with advanced reranking."""
        if not self.is_initialized:
            logger.warning("Recommendation engine not initialized")
            return []
        
        try:
            # Create user query from preferences
            user_query = self._build_user_query(preferences)
            
            # Stage 1: Initial retrieval with larger candidate set
            candidate_size = min(num_recommendations * 10, len(self.books))
            
            # Get content-based scores
            content_scores = self._get_content_scores(user_query)
            
            # Get preference-based scores
            preference_scores = self._get_preference_scores(preferences)
            
            # Get BM25 scores for better text matching
            bm25_scores = self._get_bm25_scores(user_query)
            
            # Combine initial scores (hybrid approach)
            initial_scores = self._combine_initial_scores(content_scores, preference_scores, bm25_scores)
            
            # Get top candidates for reranking
            top_candidates = self._get_top_candidates(initial_scores, candidate_size)
            
            # Stage 2: Advanced reranking
            reranked_recommendations = self._rerank_candidates(
                top_candidates, 
                preferences, 
                user_query,
                num_recommendations
            )
            
            return reranked_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _build_user_query(self, preferences: Dict) -> str:
        """Build a text query from user preferences."""
        query_parts = []
        
        # Add favorite genres
        favorite_genres = preferences.get('favorite_genres', [])
        if favorite_genres:
            query_parts.extend(favorite_genres)
        
        # Add reading mood
        reading_mood = preferences.get('reading_mood', '')
        if reading_mood:
            # Map moods to keywords
            mood_keywords = {
                'light_entertaining': ['comedy', 'humor', 'light', 'fun', 'entertaining'],
                'deep_thoughtful': ['philosophy', 'profound', 'deep', 'thoughtful', 'meaningful'],
                'escapist_adventurous': ['adventure', 'fantasy', 'escape', 'journey', 'quest'],
                'educational': ['education', 'learning', 'knowledge', 'informative'],
                'emotional': ['emotional', 'moving', 'touching', 'heartfelt'],
                'thrilling': ['thriller', 'suspense', 'exciting', 'action']
            }
            
            mood_key = reading_mood.lower().replace(' ', '_').replace('-', '_')
            if mood_key in mood_keywords:
                query_parts.extend(mood_keywords[mood_key])
        
        # Add favorite authors/books
        favorites = preferences.get('favorite_authors', '')
        if favorites:
            query_parts.append(favorites)
        
        # Add time period preferences
        time_period = preferences.get('time_period', '')
        if time_period and time_period != 'no_preference':
            if 'classic' in time_period.lower():
                query_parts.extend(['classic', 'literature', 'traditional'])
            elif 'contemporary' in time_period.lower():
                query_parts.extend(['contemporary', 'modern', 'current'])
        
        return ' '.join(query_parts)
    
    def _get_content_scores(self, user_query: str) -> np.ndarray:
        """Get content-based similarity scores using hybrid approach with compressed contexts."""
        if not user_query.strip():
            return np.zeros(len(self.books))
        
        try:
            # Get original TF-IDF scores
            user_vector = self.tfidf_vectorizer.transform([user_query])
            original_similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
            
            # Get compressed context scores
            compressed_similarities = self._get_compressed_content_scores(user_query)
            
            # Combine scores: 60% original, 40% compressed for better precision
            if compressed_similarities.max() > 0:
                # Normalize both score arrays
                original_norm = original_similarities / max(original_similarities.max(), 1e-8)
                compressed_norm = compressed_similarities / max(compressed_similarities.max(), 1e-8)
                
                # Weighted combination
                combined_similarities = 0.6 * original_norm + 0.4 * compressed_norm
                return combined_similarities
            else:
                # Fallback to original if compressed scoring fails
                return original_similarities
            
        except Exception as e:
            logger.error(f"Error calculating content scores: {e}")
            return np.zeros(len(self.books))
    
    def _get_preference_scores(self, preferences: Dict) -> np.ndarray:
        """Get preference-based scores."""
        scores = np.zeros(len(self.books))
        
        favorite_genres = preferences.get('favorite_genres', [])
        reading_mood = preferences.get('reading_mood', '')
        time_period = preferences.get('time_period', '')
        experience_level = preferences.get('experience_level', '')
        
        for i, book in enumerate(self.books):
            score = 0
            
            # Genre matching
            book_subjects = [s.lower() for s in book.get('subjects', [])]
            for genre in favorite_genres:
                genre_lower = genre.lower()
                if any(genre_lower in subject for subject in book_subjects):
                    score += 3.0
            
            # Time period matching
            if time_period and time_period != 'no_preference':
                pub_year = book.get('publication_year', '')
                if pub_year and pub_year.isdigit():
                    year = int(pub_year)
                    
                    if 'classic' in time_period.lower() and year < 1950:
                        score += 2.0
                    elif 'mid_century' in time_period.lower() and 1950 <= year <= 1980:
                        score += 2.0
                    elif 'modern' in time_period.lower() and 1980 <= year <= 2000:
                        score += 2.0
                    elif 'contemporary' in time_period.lower() and year > 2000:
                        score += 2.0
            
            # Reading mood matching
            description = book.get('description', '').lower()
            if reading_mood:
                mood_lower = reading_mood.lower()
                if 'deep' in mood_lower and any(word in description for word in ['philosophy', 'profound', 'complex']):
                    score += 1.5
                elif 'light' in mood_lower and any(word in description for word in ['humor', 'fun', 'light']):
                    score += 1.5
                elif 'adventure' in mood_lower and any(word in description for word in ['adventure', 'journey', 'quest']):
                    score += 1.5
                elif 'thrilling' in mood_lower and any(word in description for word in ['thriller', 'suspense', 'mystery']):
                    score += 1.5
            
            # Experience level matching
            if experience_level:
                if experience_level == 'literary_enthusiast':
                    if any(word in book_subjects for word in ['literature', 'classic', 'literary fiction']):
                        score += 1.0
                elif experience_level == 'casual_reader':
                    if book.get('average_rating', 0) >= 4.0:
                        score += 0.5
            
            # Rating bonus
            rating = book.get('average_rating', 0)
            if rating >= 4.5:
                score += 1.0
            elif rating >= 4.0:
                score += 0.5
            
            scores[i] = score
        
        return scores
    
    def _combine_scores(self, content_scores: np.ndarray, preference_scores: np.ndarray) -> np.ndarray:
        """Combine content and preference scores."""
        # Normalize scores to 0-1 range
        if content_scores.max() > 0:
            content_scores = content_scores / content_scores.max()
        
        if preference_scores.max() > 0:
            preference_scores = preference_scores / preference_scores.max()
        
        # Weighted combination (60% content, 40% preferences)
        final_scores = 0.6 * content_scores + 0.4 * preference_scores
        
        return final_scores
    
    def _get_top_recommendations(self, scores: np.ndarray, preferences: Dict, num_recommendations: int) -> List[Dict]:
        """Get top book recommendations with explanations."""
        # Get top indices
        top_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        seen_authors = set()
        
        for idx in top_indices:
            if len(recommendations) >= num_recommendations:
                break
            
            book = self.books[idx]
            score = scores[idx]
            
            # Skip books with very low scores
            if score < 0.1:
                continue
            
            # Avoid too many books by the same author
            authors = book.get('authors', [])
            primary_author = authors[0] if authors else 'Unknown'
            
            if primary_author not in seen_authors or len(recommendations) < num_recommendations // 2:
                recommendation = {
                    'book': book,
                    'score': float(score),
                    'reasons': self._generate_explanation(book, preferences)
                }
                
                recommendations.append(recommendation)
                seen_authors.add(primary_author)
        
        return recommendations
    
    def _generate_explanation(self, book: Dict, preferences: Dict) -> List[str]:
        """Generate explanation for why this book is recommended."""
        reasons = []
        
        favorite_genres = preferences.get('favorite_genres', [])
        book_subjects = book.get('subjects', [])
        
        # Genre matches
        matching_genres = []
        for genre in favorite_genres:
            for subject in book_subjects:
                if genre.lower() in subject.lower():
                    matching_genres.append(genre)
                    break
        
        if matching_genres:
            if len(matching_genres) == 1:
                reasons.append(f"Matches your interest in {matching_genres[0]}")
            else:
                reasons.append(f"Matches your interests in {', '.join(matching_genres)}")
        
        # Reading mood alignment
        reading_mood = preferences.get('reading_mood', '')
        description = book.get('description', '').lower()
        
        if reading_mood:
            if 'deep' in reading_mood.lower() and any(word in description for word in ['philosophy', 'profound']):
                reasons.append("Perfect for deep, thoughtful reading")
            elif 'light' in reading_mood.lower() and any(word in description for word in ['humor', 'entertaining']):
                reasons.append("Great for light, entertaining reading")
            elif 'adventure' in reading_mood.lower() and 'adventure' in description:
                reasons.append("Offers the adventurous escape you're seeking")
            elif 'thrilling' in reading_mood.lower() and any(word in description for word in ['thriller', 'suspense']):
                reasons.append("Provides the thrilling experience you want")
        
        # Time period preference
        time_period = preferences.get('time_period', '')
        pub_year = book.get('publication_year', '')
        
        if time_period and pub_year and pub_year.isdigit():
            year = int(pub_year)
            if 'classic' in time_period.lower() and year < 1950:
                reasons.append("Classic literature from your preferred era")
            elif 'contemporary' in time_period.lower() and year > 2000:
                reasons.append("Contemporary work matching your preference")
        
        # High rating
        rating = book.get('average_rating', 0)
        if rating >= 4.5:
            reasons.append("Highly rated book with excellent reviews")
        elif rating >= 4.0:
            reasons.append("Well-rated book with positive reviews")
        
        # Default reason
        if not reasons:
            reasons.append("Recommended based on your reading profile")
        
        return reasons
    
    def refresh_data(self) -> bool:
        """Refresh the recommendation engine with new data."""
        try:
            self.initialize()
            return self.is_initialized
        except Exception as e:
            logger.error(f"Error refreshing recommendation engine: {e}")
            return False
    
    def get_recommendations_by_prompt(self, prompt: str, num_recommendations: int = 3) -> List[Dict]:
        """Get book recommendations based on natural language prompt with advanced reranking."""
        if not self.is_initialized:
            logger.warning("Recommendation engine not initialized")
            return []
        
        try:
            # Clean and process the prompt
            processed_prompt = self._process_prompt(prompt)
            
            # Stage 1: Initial retrieval with larger candidate set
            candidate_size = min(num_recommendations * 10, len(self.books))
            
            # Get content-based scores using the prompt
            content_scores = self._get_content_scores(processed_prompt)
            
            # Get prompt-specific preference scores
            prompt_scores = self._get_prompt_scores(prompt)
            
            # Get BM25 scores for the original prompt
            bm25_scores = self._get_bm25_scores(prompt)
            
            # Combine initial scores
            initial_scores = self._combine_initial_scores(content_scores, prompt_scores, bm25_scores)
            
            # Get top candidates for reranking
            top_candidates = self._get_top_candidates(initial_scores, candidate_size)
            
            # Stage 2: Advanced reranking with prompt-specific preferences
            prompt_preferences = self._extract_prompt_preferences(prompt)
            reranked_recommendations = self._rerank_candidates(
                top_candidates, 
                prompt_preferences, 
                prompt,
                num_recommendations
            )
            
            # Update explanations to be prompt-specific
            for rec in reranked_recommendations:
                rec['reasons'] = self._generate_prompt_explanation(rec['book'], prompt)
            
            return reranked_recommendations
            
        except Exception as e:
            logger.error(f"Error generating prompt-based recommendations: {e}")
            return []
    
    def _process_prompt(self, prompt: str) -> str:
        """Process and enhance the user prompt for better matching."""
        # Convert to lowercase for processing
        prompt_lower = prompt.lower()
        
        # Expand common terms and synonyms
        expansions = {
            'mystery': ['mystery', 'detective', 'crime', 'investigation', 'thriller'],
            'romance': ['romance', 'love', 'relationship', 'romantic'],
            'fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'dragon', 'mythical'],
            'sci-fi': ['science fiction', 'sci-fi', 'space', 'future', 'technology', 'alien'],
            'horror': ['horror', 'scary', 'frightening', 'supernatural', 'ghost'],
            'adventure': ['adventure', 'journey', 'quest', 'exploration', 'travel'],
            'historical': ['historical', 'history', 'period', 'past', 'ancient'],
            'biography': ['biography', 'memoir', 'life story', 'autobiography'],
            'funny': ['humor', 'comedy', 'funny', 'amusing', 'hilarious'],
            'sad': ['sad', 'tragic', 'melancholy', 'emotional', 'tearjerker'],
            'war': ['war', 'military', 'battle', 'conflict', 'soldier'],
            'victorian': ['victorian', '19th century', 'england', 'british'],
            'modern': ['contemporary', 'modern', 'current', 'recent'],
            'classic': ['classic', 'literature', 'timeless', 'traditional']
        }
        
        # Expand the prompt with synonyms
        expanded_terms = []
        words = prompt_lower.split()
        
        for word in words:
            expanded_terms.append(word)
            for key, synonyms in expansions.items():
                if key in word or word in key:
                    expanded_terms.extend(synonyms)
        
        # Add the original prompt
        expanded_terms.append(prompt)
        
        return ' '.join(expanded_terms)
    
    def _get_prompt_scores(self, prompt: str) -> np.ndarray:
        """Get scores based on prompt analysis."""
        scores = np.zeros(len(self.books))
        prompt_lower = prompt.lower()
        
        # Define keyword categories and their weights
        keyword_categories = {
            'genre_keywords': {
                'mystery': ['mystery', 'detective', 'crime', 'murder', 'investigation'],
                'romance': ['romance', 'love', 'romantic', 'relationship'],
                'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical'],
                'sci-fi': ['science fiction', 'sci-fi', 'space', 'future', 'alien'],
                'horror': ['horror', 'scary', 'ghost', 'supernatural'],
                'adventure': ['adventure', 'journey', 'quest', 'exploration'],
                'historical': ['historical', 'history', 'period', 'victorian'],
                'biography': ['biography', 'memoir', 'life story'],
                'humor': ['funny', 'humor', 'comedy', 'amusing'],
                'literary': ['literary', 'literature', 'classic']
            },
            'mood_keywords': {
                'dark': ['dark', 'gritty', 'noir', 'bleak'],
                'light': ['light', 'uplifting', 'cheerful', 'positive'],
                'emotional': ['emotional', 'moving', 'touching', 'heartfelt'],
                'thrilling': ['thrilling', 'exciting', 'suspenseful', 'action'],
                'thoughtful': ['thoughtful', 'philosophical', 'deep', 'profound']
            },
            'setting_keywords': {
                'london': ['london', 'british', 'england'],
                'victorian': ['victorian', '19th century'],
                'modern': ['modern', 'contemporary', 'current'],
                'space': ['space', 'galaxy', 'universe', 'planet'],
                'medieval': ['medieval', 'middle ages', 'knights']
            }
        }
        
        for i, book in enumerate(self.books):
            score = 0
            
            # Get book text for matching
            book_text = ' '.join([
                book.get('title', ''),
                ' '.join(book.get('authors', [])),
                ' '.join(book.get('subjects', [])),
                book.get('description', ''),
                book.get('publisher', '')
            ]).lower()
            
            # Score based on keyword categories
            for category, subcategories in keyword_categories.items():
                for subcat, keywords in subcategories.items():
                    prompt_matches = sum(1 for keyword in keywords if keyword in prompt_lower)
                    book_matches = sum(1 for keyword in keywords if keyword in book_text)
                    
                    if prompt_matches > 0 and book_matches > 0:
                        # Higher score for more matches
                        match_score = min(prompt_matches, book_matches) * 2.0
                        if category == 'genre_keywords':
                            score += match_score * 1.5  # Genre matches are more important
                        else:
                            score += match_score
            
            # Direct text similarity bonus
            prompt_words = set(prompt_lower.split())
            book_words = set(book_text.split())
            common_words = prompt_words.intersection(book_words)
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            meaningful_common = common_words - stop_words
            
            if meaningful_common:
                score += len(meaningful_common) * 0.5
            
            # Rating bonus
            rating = book.get('average_rating', 0)
            if rating >= 4.0:
                score += 0.5
            
            scores[i] = score
        
        return scores
    
    def _get_top_prompt_recommendations(self, scores: np.ndarray, prompt: str, num_recommendations: int) -> List[Dict]:
        """Get top book recommendations with prompt-specific explanations."""
        # Get top indices
        top_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        seen_authors = set()
        
        for idx in top_indices:
            if len(recommendations) >= num_recommendations:
                break
            
            book = self.books[idx]
            score = scores[idx]
            
            # Skip books with very low scores
            if score < 0.1:
                continue
            
            # Avoid too many books by the same author
            authors = book.get('authors', [])
            primary_author = authors[0] if authors else 'Unknown'
            
            if primary_author not in seen_authors or len(recommendations) < num_recommendations // 2:
                recommendation = {
                    'book': book,
                    'score': float(score),
                    'reasons': self._generate_prompt_explanation(book, prompt)
                }
                
                recommendations.append(recommendation)
                seen_authors.add(primary_author)
        
        return recommendations
    
    def _generate_prompt_explanation(self, book: Dict, prompt: str) -> List[str]:
        """Generate explanation for why this book matches the prompt."""
        reasons = []
        prompt_lower = prompt.lower()
        
        # Get book information
        title = book.get('title', '').lower()
        subjects = [s.lower() for s in book.get('subjects', [])]
        description = book.get('description', '').lower()
        authors = book.get('authors', [])
        
        # Check for direct matches in different categories
        genre_matches = []
        mood_matches = []
        setting_matches = []
        
        # Genre matching - focused on Science, Technology, and AI
        genre_keywords = {
            'artificial intelligence': ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural networks'],
            'computer science': ['computer science', 'programming', 'software engineering', 'algorithms'],
            'data science': ['data science', 'big data', 'data analysis', 'statistics', 'analytics'],
            'robotics': ['robotics', 'automation', 'robot', 'autonomous'],
            'technology': ['technology', 'tech', 'innovation', 'digital', 'computing'],
            'engineering': ['engineering', 'electrical', 'mechanical', 'bioengineering', 'aerospace'],
            'physics': ['physics', 'quantum', 'astrophysics', 'theoretical physics'],
            'chemistry': ['chemistry', 'biochemistry', 'chemical engineering', 'materials science'],
            'biology': ['biology', 'biotechnology', 'genetics', 'genomics', 'bioinformatics'],
            'mathematics': ['mathematics', 'math', 'calculus', 'statistics', 'mathematical'],
            'cybersecurity': ['cybersecurity', 'security', 'cryptography', 'hacking', 'cyber'],
            'space science': ['space', 'astronomy', 'astrophysics', 'cosmology', 'planetary']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                if any(keyword in ' '.join(subjects) or keyword in description or keyword in title for keyword in keywords):
                    genre_matches.append(genre)
        
        if genre_matches:
            reasons.append(f"Matches your interest in {', '.join(genre_matches)}")
        
        # Setting/time period matching
        if 'victorian' in prompt_lower and any('victorian' in s or '19th' in s for s in subjects):
            reasons.append("Set in the Victorian era as requested")
        elif 'london' in prompt_lower and any('london' in s or 'british' in s for s in subjects + [description]):
            reasons.append("Takes place in London as you wanted")
        elif 'space' in prompt_lower and any('space' in s or 'science fiction' in s for s in subjects):
            reasons.append("Features the space setting you're looking for")
        
        # Mood matching
        if 'thrilling' in prompt_lower and any(word in description for word in ['thriller', 'suspense', 'exciting']):
            reasons.append("Provides the thrilling experience you want")
        elif 'heartwarming' in prompt_lower and any(word in description for word in ['heartwarming', 'touching', 'emotional']):
            reasons.append("Offers the heartwarming story you're seeking")
        elif 'funny' in prompt_lower and any(word in description for word in ['humor', 'funny', 'comedy']):
            reasons.append("Contains the humor you're looking for")
        
        # Author matching
        if authors:
            author_names = ' '.join(authors).lower()
            prompt_words = prompt_lower.split()
            for word in prompt_words:
                if len(word) > 3 and word in author_names:
                    reasons.append(f"Written by {authors[0]} as mentioned in your request")
                    break
        
        # High rating
        rating = book.get('average_rating', 0)
        if rating >= 4.5:
            reasons.append("Highly rated with excellent reviews")
        elif rating >= 4.0:
            reasons.append("Well-reviewed by readers")
        
        # Default reason if no specific matches
        if not reasons:
            reasons.append("Closely matches your description")
        
        return reasons

    def get_similar_books(self, book_id: str, num_similar: int = 5) -> List[Dict]:
        """Get books similar to a specific book."""
        if not self.is_initialized:
            return []
        
        try:
            # Find the book index
            book_index = None
            for i, book in enumerate(self.books):
                if book.get('id') == book_id:
                    book_index = i
                    break
            
            if book_index is None:
                return []
            
            # Get similarity scores for this book
            book_vector = self.tfidf_matrix[book_index]
            similarities = cosine_similarity(book_vector, self.tfidf_matrix).flatten()
            
            # Get top similar books (excluding the book itself)
            similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]
            
            similar_books = []
            for idx in similar_indices:
                similar_books.append({
                    'book': self.books[idx],
                    'similarity_score': float(similarities[idx])
                })
            
            return similar_books
            
        except Exception as e:
            logger.error(f"Error finding similar books: {e}")
            return []

    # ==================== ADVANCED RERANKING METHODS ====================
    
    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for better text matching."""
        if not query.strip():
            return np.zeros(len(self.books))
        
        try:
            # BM25 parameters
            k1 = 1.5  # Term frequency saturation parameter
            b = 0.75  # Length normalization parameter
            
            # Tokenize query
            query_terms = query.lower().split()
            
            # Calculate document lengths
            doc_lengths = []
            all_docs = []
            
            for book in self.books:
                doc_text = ' '.join([
                    book.get('title', ''),
                    ' '.join(book.get('authors', [])),
                    ' '.join(book.get('subjects', [])),
                    book.get('description', ''),
                ]).lower()
                
                doc_tokens = doc_text.split()
                doc_lengths.append(len(doc_tokens))
                all_docs.append(doc_tokens)
            
            avg_doc_length = np.mean(doc_lengths) if doc_lengths else 1
            
            # Calculate BM25 scores
            scores = np.zeros(len(self.books))
            
            for i, doc_tokens in enumerate(all_docs):
                doc_length = doc_lengths[i]
                score = 0
                
                for term in query_terms:
                    # Term frequency in document
                    tf = doc_tokens.count(term)
                    
                    if tf > 0:
                        # Document frequency (number of documents containing term)
                        df = sum(1 for doc in all_docs if term in doc)
                        
                        # Inverse document frequency
                        idf = math.log((len(self.books) - df + 0.5) / (df + 0.5))
                        
                        # BM25 score component
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                        
                        score += idf * (numerator / denominator)
                
                scores[i] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating BM25 scores: {e}")
            return np.zeros(len(self.books))
    
    def _combine_initial_scores(self, content_scores: np.ndarray, preference_scores: np.ndarray, bm25_scores: np.ndarray) -> np.ndarray:
        """Combine initial retrieval scores."""
        # Normalize all score arrays
        def normalize_scores(scores):
            if scores.max() > 0:
                return scores / scores.max()
            return scores
        
        content_norm = normalize_scores(content_scores)
        preference_norm = normalize_scores(preference_scores)
        bm25_norm = normalize_scores(bm25_scores)
        
        # Weighted combination: 40% content, 30% preferences, 30% BM25
        combined_scores = 0.4 * content_norm + 0.3 * preference_norm + 0.3 * bm25_norm
        
        return combined_scores
    
    def _get_top_candidates(self, scores: np.ndarray, candidate_size: int) -> List[Tuple[int, float]]:
        """Get top candidates for reranking."""
        top_indices = np.argsort(scores)[::-1][:candidate_size]
        candidates = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        return candidates
    
    def _rerank_candidates(self, candidates: List[Tuple[int, float]], preferences: Dict, query: str, num_recommendations: int) -> List[Dict]:
        """Advanced reranking of candidate books."""
        if not candidates:
            return []
        
        # Extract candidate books and their initial scores
        candidate_books = []
        for idx, initial_score in candidates:
            book = self.books[idx]
            candidate_books.append({
                'book': book,
                'index': idx,
                'initial_score': initial_score
            })
        
        # Calculate reranking features for each candidate
        reranked_candidates = []
        for candidate in candidate_books:
            book = candidate['book']
            features = self._extract_reranking_features(book, preferences, query)
            
            # Calculate final reranking score
            rerank_score = self._calculate_rerank_score(features, candidate['initial_score'])
            
            reranked_candidates.append({
                'book': book,
                'initial_score': candidate['initial_score'],
                'rerank_score': rerank_score,
                'features': features
            })
        
        # Sort by reranking score
        reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply diversity and quality filters
        final_recommendations = self._apply_diversity_filters(
            reranked_candidates, 
            preferences, 
            num_recommendations
        )
        
        return final_recommendations

    # ==================== CONTEXT COMPRESSION METHODS ====================
    
    def _create_compressed_contexts(self) -> None:
        """Create compressed contexts for all books using extractive summarization and semantic compression."""
        if not self.books:
            return
        
        self.compressed_books = []
        
        for book in self.books:
            compressed_book = self._compress_book_context(book)
            self.compressed_books.append(compressed_book)
        
        logger.info(f"Created compressed contexts for {len(self.compressed_books)} books")
    
    def _compress_book_context(self, book: Dict) -> Dict:
        """Compress a single book's context using multiple compression techniques."""
        # Extract key information
        title = book.get('title', '')
        authors = book.get('authors', [])
        subjects = book.get('subjects', [])
        description = book.get('description', '')
        publisher = book.get('publisher', '')
        
        # Compress description using extractive summarization
        compressed_description = self._extract_key_sentences(description)
        
        # Extract key phrases and themes
        key_phrases = self._extract_key_phrases(description, subjects)
        
        # Create genre-specific keywords
        genre_keywords = self._extract_genre_keywords(subjects, description)
        
        # Create thematic elements
        themes = self._extract_themes(description, title)
        
        # Build compressed context
        compressed_context_parts = []
        
        # Add title with high weight
        if title:
            compressed_context_parts.extend([title] * 3)
        
        # Add authors
        if authors:
            compressed_context_parts.extend(authors)
        
        # Add compressed subjects/genres
        if subjects:
            compressed_subjects = self._compress_subjects(subjects)
            compressed_context_parts.extend(compressed_subjects)
        
        # Add genre-specific keywords
        if genre_keywords:
            compressed_context_parts.extend(genre_keywords)
        
        # Add key phrases
        if key_phrases:
            compressed_context_parts.extend(key_phrases)
        
        # Add themes
        if themes:
            compressed_context_parts.extend(themes)
        
        # Add compressed description
        if compressed_description:
            compressed_context_parts.append(compressed_description)
        
        # Add publisher if relevant
        if publisher and len(publisher.split()) <= 3:  # Only short publisher names
            compressed_context_parts.append(publisher)
        
        # Create the compressed book object
        compressed_book = {
            'id': book.get('id'),
            'title': title,
            'authors': authors,
            'compressed_context': ' '.join(compressed_context_parts),
            'key_phrases': key_phrases,
            'themes': themes,
            'genre_keywords': genre_keywords,
            'compressed_description': compressed_description
        }
        
        return compressed_book
    
    def _extract_key_sentences(self, description: str, max_sentences: int = 2) -> str:
        """Extract key sentences from description using simple extractive summarization."""
        if not description or len(description) < 100:
            return description
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', description)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return description
        
        # Score sentences based on various factors
        sentence_scores = []
        
        # Important keywords that indicate key information
        important_keywords = [
            'story', 'tale', 'follows', 'about', 'chronicles', 'explores',
            'journey', 'adventure', 'mystery', 'romance', 'thriller',
            'historical', 'fantasy', 'science fiction', 'biography',
            'war', 'love', 'family', 'friendship', 'betrayal', 'murder',
            'detective', 'investigation', 'magic', 'wizard', 'dragon',
            'space', 'future', 'past', 'victorian', 'medieval'
        ]
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on important keywords
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Prefer sentences with specific details
            if any(word in sentence_lower for word in ['when', 'where', 'who', 'what', 'how']):
                score += 1
            
            # Prefer sentences that are not too short or too long
            word_count = len(sentence.split())
            if 10 <= word_count <= 25:
                score += 1
            
            # Prefer sentences near the beginning
            sentence_position = sentences.index(sentence)
            if sentence_position < len(sentences) // 3:
                score += 1
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        # Maintain original order
        result_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                result_sentences.append(sentence)
        
        return '. '.join(result_sentences) + '.' if result_sentences else description[:200]
    
    def _extract_key_phrases(self, description: str, subjects: List[str]) -> List[str]:
        """Extract key phrases that capture the essence of the book."""
        if not description:
            return []
        
        key_phrases = []
        description_lower = description.lower()
        
        # Define phrase patterns that are often important
        phrase_patterns = [
            # Character types
            r'\b(?:young|old|mysterious|brave|cunning|wise|beautiful|handsome)\s+(?:woman|man|girl|boy|detective|wizard|knight|princess|prince|hero|heroine)\b',
            
            # Settings and time periods
            r'\b(?:in|during|set in)\s+(?:victorian|medieval|ancient|modern|contemporary|19th century|20th century)\b',
            r'\b(?:london|paris|new york|england|france|america|scotland|ireland|italy|spain)\b',
            
            # Plot elements
            r'\b(?:murder|mystery|investigation|romance|love story|adventure|quest|journey|war|battle|magic|spell|curse)\b',
            
            # Emotional themes
            r'\b(?:heartwarming|thrilling|suspenseful|romantic|tragic|comedic|humorous|dark|gritty|uplifting)\b',
            
            # Literary elements
            r'\b(?:coming of age|family saga|historical fiction|science fiction|fantasy|biography|memoir|autobiography)\b'
        ]
        
        # Extract phrases using patterns
        for pattern in phrase_patterns:
            matches = re.findall(pattern, description_lower)
            key_phrases.extend(matches)
        
        # Add subject-based phrases
        for subject in subjects:
            subject_lower = subject.lower()
            if len(subject_lower.split()) <= 3:  # Only short, meaningful subjects
                key_phrases.append(subject_lower)
        
        # Remove duplicates and return unique phrases
        unique_phrases = list(set(key_phrases))
        return unique_phrases[:10]  # Limit to top 10 phrases
    
    def _extract_genre_keywords(self, subjects: List[str], description: str) -> List[str]:
        """Extract genre-specific keywords that help with categorization."""
        genre_keywords = []
        
        # Genre-specific keyword mappings
        genre_mappings = {
            'mystery': ['detective', 'investigation', 'murder', 'crime', 'clues', 'suspect', 'solve'],
            'romance': ['love', 'romantic', 'relationship', 'marriage', 'passion', 'heart'],
            'fantasy': ['magic', 'wizard', 'dragon', 'quest', 'kingdom', 'spell', 'enchanted'],
            'science fiction': ['space', 'future', 'technology', 'alien', 'robot', 'time travel'],
            'historical': ['historical', 'period', 'era', 'century', 'past', 'ancient'],
            'thriller': ['suspense', 'tension', 'danger', 'chase', 'escape', 'threat'],
            'horror': ['scary', 'frightening', 'ghost', 'supernatural', 'haunted', 'terror'],
            'adventure': ['journey', 'exploration', 'discovery', 'expedition', 'travel'],
            'biography': ['life', 'biography', 'memoir', 'personal', 'real', 'true story'],
            'war': ['war', 'battle', 'military', 'soldier', 'conflict', 'combat']
        }
        
        description_lower = description.lower() if description else ''
        subjects_text = ' '.join(subjects).lower()
        
        # Check each genre and extract relevant keywords
        for genre, keywords in genre_mappings.items():
            if any(genre_word in subjects_text for genre_word in [genre, genre.replace(' ', '')]):
                # Add keywords that appear in the description
                for keyword in keywords:
                    if keyword in description_lower:
                        genre_keywords.append(keyword)
        
        return list(set(genre_keywords))[:8]  # Limit and remove duplicates
    
    def _extract_themes(self, description: str, title: str) -> List[str]:
        """Extract thematic elements from the book."""
        if not description:
            return []
        
        themes = []
        text = (description + ' ' + title).lower()
        
        # Define thematic categories
        theme_keywords = {
            'family': ['family', 'mother', 'father', 'sister', 'brother', 'daughter', 'son', 'parent'],
            'friendship': ['friend', 'friendship', 'companion', 'ally', 'bond'],
            'love': ['love', 'romance', 'relationship', 'marriage', 'wedding', 'couple'],
            'betrayal': ['betrayal', 'betray', 'deceive', 'lie', 'cheat', 'unfaithful'],
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution'],
            'power': ['power', 'control', 'authority', 'rule', 'command', 'dominance'],
            'justice': ['justice', 'law', 'court', 'judge', 'trial', 'verdict'],
            'survival': ['survival', 'survive', 'escape', 'flee', 'refuge'],
            'identity': ['identity', 'self', 'who am i', 'discover', 'find yourself'],
            'sacrifice': ['sacrifice', 'give up', 'loss', 'cost', 'price']
        }
        
        # Check for themes
        for theme, keywords in theme_keywords.items():
            if any(keyword in text for keyword in keywords):
                themes.append(theme)
        
        return themes[:5]  # Limit to top 5 themes
    
    def _compress_subjects(self, subjects: List[str]) -> List[str]:
        """Compress and clean subject/genre information."""
        if not subjects:
            return []
        
        compressed_subjects = []
        
        # Priority subjects (more important genres/topics)
        priority_keywords = [
            'fiction', 'mystery', 'romance', 'fantasy', 'science fiction',
            'historical', 'biography', 'thriller', 'horror', 'adventure',
            'literary', 'classic', 'contemporary', 'young adult', 'children'
        ]
        
        # First, add priority subjects
        for subject in subjects:
            subject_lower = subject.lower()
            for priority in priority_keywords:
                if priority in subject_lower and len(subject.split()) <= 4:
                    compressed_subjects.append(subject)
                    break
        
        # Then add other short, meaningful subjects
        for subject in subjects:
            if (len(subject.split()) <= 3 and 
                subject not in compressed_subjects and 
                len(compressed_subjects) < 8):
                compressed_subjects.append(subject)
        
        return compressed_subjects
    
    def _get_compressed_content_scores(self, user_query: str) -> np.ndarray:
        """Get content-based similarity scores using compressed contexts."""
        if not user_query.strip() or self.compressed_tfidf_matrix is None:
            return np.zeros(len(self.books))
        
        try:
            # Transform user query using the compressed vectorizer
            # Note: We need to store the compressed vectorizer as an instance variable
            if not hasattr(self, 'compressed_vectorizer'):
                return self._get_content_scores(user_query)  # Fallback to original method
            
            user_vector = self.compressed_vectorizer.transform([user_query])
            
            # Calculate cosine similarity with compressed contexts
            similarities = cosine_similarity(user_vector, self.compressed_tfidf_matrix).flatten()
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating compressed content scores: {e}")
            return self._get_content_scores(user_query)  # Fallback to original method
    def _extract_prompt_preferences(self, prompt: str) -> Dict:
        """Extract preferences from natural language prompt."""
        prompt_lower = prompt.lower()
        preferences = {
            'favorite_genres': [],
            'reading_mood': '',
            'time_period': 'no_preference',
            'experience_level': 'regular_reader'
        }
        
        # Extract genres from prompt
        genre_mapping = {
            'mystery': ['mystery', 'detective', 'crime', 'murder', 'investigation'],
            'romance': ['romance', 'love', 'romantic', 'relationship'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical'],
            'science fiction': ['sci-fi', 'science fiction', 'space', 'future', 'alien'],
            'horror': ['horror', 'scary', 'ghost', 'supernatural'],
            'adventure': ['adventure', 'journey', 'quest', 'exploration'],
            'historical': ['historical', 'history', 'period', 'victorian'],
            'biography': ['biography', 'memoir', 'life story'],
            'humor': ['funny', 'humor', 'comedy', 'amusing'],
            'literary': ['literary', 'literature', 'classic']
        }
        
        for genre, keywords in genre_mapping.items():
            if any(keyword in prompt_lower for keyword in keywords):
                preferences['favorite_genres'].append(genre.title())
        
        # Extract reading mood
        if any(word in prompt_lower for word in ['thrilling', 'exciting', 'suspenseful', 'action']):
            preferences['reading_mood'] = 'thrilling'
        elif any(word in prompt_lower for word in ['light', 'funny', 'humor', 'amusing', 'entertaining']):
            preferences['reading_mood'] = 'light_entertaining'
        elif any(word in prompt_lower for word in ['deep', 'thoughtful', 'philosophical', 'profound']):
            preferences['reading_mood'] = 'deep_thoughtful'
        elif any(word in prompt_lower for word in ['adventure', 'escape', 'journey', 'quest']):
            preferences['reading_mood'] = 'escapist_adventurous'
        elif any(word in prompt_lower for word in ['emotional', 'moving', 'touching', 'heartfelt']):
            preferences['reading_mood'] = 'emotional'
        elif any(word in prompt_lower for word in ['educational', 'learning', 'informative']):
            preferences['reading_mood'] = 'educational'
        
        # Extract time period preferences
        if any(word in prompt_lower for word in ['classic', 'traditional', 'old']):
            preferences['time_period'] = 'classic'
        elif any(word in prompt_lower for word in ['contemporary', 'modern', 'current', 'recent']):
            preferences['time_period'] = 'contemporary'
        elif any(word in prompt_lower for word in ['victorian', '19th century']):
            preferences['time_period'] = 'classic'
        
        # Extract experience level
        if any(word in prompt_lower for word in ['literary', 'literature', 'sophisticated', 'complex']):
            preferences['experience_level'] = 'literary_enthusiast'
        elif any(word in prompt_lower for word in ['easy', 'simple', 'casual', 'light reading']):
            preferences['experience_level'] = 'casual_reader'
        
        return preferences
    
    def _extract_reranking_features(self, book: Dict, preferences: Dict, query: str) -> Dict:
        """Extract features for reranking."""
        features = {}
        
        # Basic book features
        features['rating'] = book.get('average_rating', 0)
        features['rating_count'] = book.get('ratings_count', 0)
        features['page_count'] = book.get('page_count', 0)
        
        # Publication recency (higher score for newer books, but not too much weight)
        pub_year = book.get('publication_year', '')
        if pub_year and pub_year.isdigit():
            current_year = datetime.now().year
            year_diff = current_year - int(pub_year)
            features['recency'] = max(0, 1 - (year_diff / 100))  # Normalize to 0-1
        else:
            features['recency'] = 0.5  # Neutral score for unknown dates
        
        # Genre alignment
        book_subjects = [s.lower() for s in book.get('subjects', [])]
        favorite_genres = [g.lower() for g in preferences.get('favorite_genres', [])]
        
        genre_matches = sum(1 for genre in favorite_genres 
                           for subject in book_subjects 
                           if genre in subject)
        features['genre_alignment'] = min(1.0, genre_matches / max(1, len(favorite_genres)))
        
        # Query relevance (exact term matches)
        query_terms = set(query.lower().split())
        book_text = ' '.join([
            book.get('title', ''),
            ' '.join(book.get('authors', [])),
            ' '.join(book.get('subjects', [])),
            book.get('description', '')[:200]  # First 200 chars of description
        ]).lower()
        
        book_terms = set(book_text.split())
        common_terms = query_terms.intersection(book_terms)
        features['query_relevance'] = len(common_terms) / max(1, len(query_terms))
        
        # Title relevance (higher weight for title matches)
        title_terms = set(book.get('title', '').lower().split())
        title_matches = query_terms.intersection(title_terms)
        features['title_relevance'] = len(title_matches) / max(1, len(query_terms))
        
        # Author popularity (based on number of books by same author)
        authors = book.get('authors', [])
        if authors:
            author_book_count = sum(1 for b in self.books 
                                  if any(author in b.get('authors', []) for author in authors))
            features['author_popularity'] = min(1.0, author_book_count / 10)  # Normalize
        else:
            features['author_popularity'] = 0
        
        # Description quality (longer descriptions might be more informative)
        description = book.get('description', '')
        features['description_quality'] = min(1.0, len(description) / 500)  # Normalize to 500 chars
        
        # Reading level match
        experience_level = preferences.get('experience_level', '')
        if experience_level == 'literary_enthusiast':
            literary_indicators = ['literature', 'classic', 'literary fiction', 'poetry']
            features['reading_level_match'] = 1.0 if any(ind in ' '.join(book_subjects) for ind in literary_indicators) else 0.5
        elif experience_level == 'casual_reader':
            features['reading_level_match'] = 1.0 if features['rating'] >= 4.0 else 0.7
        else:
            features['reading_level_match'] = 0.8  # Neutral
        
        return features
    
    def _calculate_rerank_score(self, features: Dict, initial_score: float) -> float:
        """Calculate final reranking score using feature weights."""
        # Feature weights (tuned for book recommendations)
        weights = {
            'rating': 0.20,
            'rating_count': 0.05,
            'genre_alignment': 0.25,
            'query_relevance': 0.20,
            'title_relevance': 0.15,
            'author_popularity': 0.05,
            'description_quality': 0.05,
            'reading_level_match': 0.10,
            'recency': 0.05
        }
        
        # Calculate weighted feature score
        feature_score = 0
        for feature, value in features.items():
            if feature in weights:
                # Apply logarithmic scaling for rating_count to prevent dominance
                if feature == 'rating_count':
                    value = math.log(value + 1) / math.log(1000)  # Normalize to log scale
                    value = min(1.0, value)
                
                feature_score += weights[feature] * value
        
        # Combine with initial score (70% features, 30% initial retrieval score)
        final_score = 0.7 * feature_score + 0.3 * initial_score
        
        return final_score
    
    def _apply_diversity_filters(self, candidates: List[Dict], preferences: Dict, num_recommendations: int) -> List[Dict]:
        """Apply diversity and quality filters to final recommendations."""
        if not candidates:
            return []
        
        final_recommendations = []
        seen_authors = set()
        seen_genres = set()
        
        # Sort candidates by rerank score
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        for candidate in candidates:
            if len(final_recommendations) >= num_recommendations:
                break
            
            book = candidate['book']
            authors = book.get('authors', [])
            subjects = book.get('subjects', [])
            
            # Author diversity: limit books per author
            primary_author = authors[0] if authors else 'Unknown'
            author_count = sum(1 for rec in final_recommendations 
                             if rec['book'].get('authors', [''])[0] == primary_author)
            
            # Genre diversity: ensure variety in genres
            book_genres = set(s.lower() for s in subjects)
            genre_overlap = len(book_genres.intersection(seen_genres))
            
            # Quality threshold: minimum score requirement
            min_score_threshold = 0.3
            
            # Diversity scoring
            diversity_penalty = 0
            if author_count >= 1:  # Already have a book by this author
                diversity_penalty += 0.2
            if genre_overlap > 2:  # Too much genre overlap
                diversity_penalty += 0.1
            
            adjusted_score = candidate['rerank_score'] - diversity_penalty
            
            # Accept if meets criteria
            if (adjusted_score >= min_score_threshold and 
                (author_count == 0 or len(final_recommendations) < num_recommendations // 2)):
                
                recommendation = {
                    'book': book,
                    'score': float(adjusted_score),
                    'reasons': self._generate_explanation(book, preferences),
                    'rerank_features': candidate['features']
                }
                
                final_recommendations.append(recommendation)
                seen_authors.add(primary_author)
                seen_genres.update(book_genres)
        
        # If we don't have enough recommendations, fill with top candidates
        while len(final_recommendations) < num_recommendations and len(final_recommendations) < len(candidates):
            for candidate in candidates:
                if len(final_recommendations) >= num_recommendations:
                    break
                
                book = candidate['book']
                if not any(rec['book'].get('id') == book.get('id') for rec in final_recommendations):
                    recommendation = {
                        'book': book,
                        'score': float(candidate['rerank_score']),
                        'reasons': self._generate_explanation(book, preferences),
                        'rerank_features': candidate['features']
                    }
                    final_recommendations.append(recommendation)
        
        return final_recommendations

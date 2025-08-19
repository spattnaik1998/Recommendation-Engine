#!/usr/bin/env python3
"""
RAG-Based Book Recommendation Engine
Uses user preferences and reading history to provide personalized book recommendations.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')

class BookRecommenderRAG:
    def __init__(self, books_file: str = 'data/books_database.json'):
        """Initialize the RAG-based book recommender."""
        self.books_file = books_file
        self.books = []
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.book_features = None
        self.user_profile = {}
        
    def load_books(self) -> bool:
        """Load book data from JSON file."""
        try:
            with open(self.books_file, 'r', encoding='utf-8') as f:
                self.books = json.load(f)
            
            if not self.books:
                print(f"No books found in {self.books_file}")
                return False
                
            print(f"📚 Loaded {len(self.books)} books for recommendation engine")
            self._prepare_book_data()
            return True
            
        except FileNotFoundError:
            print(f"❌ File {self.books_file} not found!")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _prepare_book_data(self):
        """Prepare book data for recommendation processing."""
        # Create DataFrame
        data = []
        for i, book in enumerate(self.books):
            # Combine text features for content-based filtering
            text_features = []
            text_features.extend(book.get('subjects', []))
            text_features.append(book.get('description', ''))
            text_features.extend(book.get('authors', []))
            
            # Extract publication decade
            pub_year = self._extract_year(book.get('first_publish_date', ''))
            pub_decade = (pub_year // 10) * 10 if pub_year > 0 else 0
            
            data.append({
                'book_id': i,
                'title': book.get('title', 'Unknown'),
                'authors': book.get('authors', ['Unknown']),
                'subjects': book.get('subjects', []),
                'description': book.get('description', ''),
                'publication_year': pub_year,
                'publication_decade': pub_decade,
                'text_features': ' '.join(text_features),
                'has_cover': len(book.get('covers', [])) > 0
            })
        
        self.df = pd.DataFrame(data)
        
        # Create TF-IDF matrix for content-based recommendations
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['text_features'])
        
        # Create additional feature matrix
        self._create_feature_matrix()
        
        print("✅ Book data prepared for recommendations")
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from publication date string."""
        if not date_str:
            return 0
        
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(date_str))
        if year_match:
            return int(year_match.group(1))
        return 0
    
    def _create_feature_matrix(self):
        """Create additional feature matrix for hybrid recommendations."""
        features = []
        
        for _, book in self.df.iterrows():
            feature_vector = []
            
            # Publication decade (normalized)
            decade_features = [0] * 20  # Cover 1800s to 2000s
            if book['publication_decade'] >= 1800:
                decade_idx = min((book['publication_decade'] - 1800) // 10, 19)
                decade_features[decade_idx] = 1
            
            # Subject categories
            subject_categories = [
                'fiction', 'fantasy', 'science fiction', 'romance', 'mystery',
                'adventure', 'philosophy', 'literature', 'historical', 'children'
            ]
            
            subject_features = []
            subjects_lower = [s.lower() for s in book['subjects']]
            for category in subject_categories:
                has_category = any(category in subject for subject in subjects_lower)
                subject_features.append(1 if has_category else 0)
            
            # Combine features
            feature_vector.extend(decade_features)
            feature_vector.extend(subject_features)
            feature_vector.append(1 if book['has_cover'] else 0)
            
            features.append(feature_vector)
        
        self.book_features = np.array(features)
    
    def collect_user_preferences(self) -> Dict:
        """Collect user preferences through interactive questions."""
        print("\n" + "="*60)
        print("📋 BOOK RECOMMENDATION QUESTIONNAIRE")
        print("="*60)
        print("Please answer the following questions to get personalized recommendations:")
        
        preferences = {}
        
        # Favorite genres
        print("\n1. What are your favorite genres? (Select multiple by number, separated by commas)")
        genres = [
            "Fiction", "Fantasy", "Science Fiction", "Romance", "Mystery/Crime",
            "Adventure", "Philosophy", "Classic Literature", "Historical Fiction",
            "Children's/Young Adult", "Biography", "Non-fiction", "Horror",
            "Thriller", "Comedy/Humor"
        ]
        
        for i, genre in enumerate(genres, 1):
            print(f"   {i:2d}. {genre}")
        
        try:
            genre_input = input("\nYour choices (e.g., 1,3,7): ").strip()
            selected_genres = []
            if genre_input:
                for num in genre_input.split(','):
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(genres):
                        selected_genres.append(genres[idx])
            preferences['favorite_genres'] = selected_genres
        except:
            preferences['favorite_genres'] = []
        
        # Reading mood
        print("\n2. What's your current reading mood?")
        moods = [
            "Light and entertaining", "Deep and thought-provoking",
            "Escapist and adventurous", "Educational and informative",
            "Emotional and moving", "Fast-paced and thrilling"
        ]
        
        for i, mood in enumerate(moods, 1):
            print(f"   {i}. {mood}")
        
        try:
            mood_choice = int(input("\nYour choice (1-6): ").strip())
            preferences['reading_mood'] = moods[mood_choice - 1] if 1 <= mood_choice <= len(moods) else ""
        except:
            preferences['reading_mood'] = ""
        
        # Time period preference
        print("\n3. Do you prefer books from a specific time period?")
        periods = [
            "No preference", "Classic (before 1950)", "Mid-century (1950-1980)",
            "Modern (1980-2000)", "Contemporary (2000+)", "Historical settings"
        ]
        
        for i, period in enumerate(periods, 1):
            print(f"   {i}. {period}")
        
        try:
            period_choice = int(input("\nYour choice (1-6): ").strip())
            preferences['time_period'] = periods[period_choice - 1] if 1 <= period_choice <= len(periods) else "No preference"
        except:
            preferences['time_period'] = "No preference"
        
        # Book length preference
        print("\n4. What's your preference for book length/complexity?")
        lengths = [
            "Short and concise", "Medium length", "Long and detailed",
            "No preference"
        ]
        
        for i, length in enumerate(lengths, 1):
            print(f"   {i}. {length}")
        
        try:
            length_choice = int(input("\nYour choice (1-4): ").strip())
            preferences['book_length'] = lengths[length_choice - 1] if 1 <= length_choice <= len(lengths) else "No preference"
        except:
            preferences['book_length'] = "No preference"
        
        # Favorite authors or similar books
        print("\n5. Do you have any favorite authors or books you've enjoyed recently?")
        print("   (This helps us find similar recommendations)")
        
        try:
            favorites = input("Enter names (optional): ").strip()
            preferences['favorites'] = favorites
        except:
            preferences['favorites'] = ""
        
        # Reading experience level
        print("\n6. How would you describe your reading experience?")
        experience_levels = [
            "Casual reader", "Regular reader", "Avid reader", "Literary enthusiast"
        ]
        
        for i, level in enumerate(experience_levels, 1):
            print(f"   {i}. {level}")
        
        try:
            exp_choice = int(input("\nYour choice (1-4): ").strip())
            preferences['experience_level'] = experience_levels[exp_choice - 1] if 1 <= exp_choice <= len(experience_levels) else "Regular reader"
        except:
            preferences['experience_level'] = "Regular reader"
        
        self.user_profile = preferences
        print("\n✅ Preferences collected! Generating recommendations...")
        return preferences
    
    def _calculate_content_similarity(self, user_query: str) -> np.ndarray:
        """Calculate content-based similarity scores."""
        if not user_query.strip():
            return np.zeros(len(self.books))
        
        # Transform user query using the same TF-IDF vectorizer
        user_vector = self.tfidf_vectorizer.transform([user_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        return similarities
    
    def _calculate_preference_scores(self) -> np.ndarray:
        """Calculate preference-based scores for each book."""
        scores = np.zeros(len(self.books))
        
        if not self.user_profile:
            return scores
        
        for i, book in enumerate(self.books):
            score = 0
            
            # Genre preferences
            favorite_genres = self.user_profile.get('favorite_genres', [])
            book_subjects = [s.lower() for s in book.get('subjects', [])]
            
            for genre in favorite_genres:
                genre_lower = genre.lower()
                if any(genre_lower in subject for subject in book_subjects):
                    score += 2.0
            
            # Time period preferences
            time_pref = self.user_profile.get('time_period', '')
            pub_year = self._extract_year(book.get('first_publish_date', ''))
            
            if time_pref == "Classic (before 1950)" and pub_year < 1950 and pub_year > 0:
                score += 1.5
            elif time_pref == "Mid-century (1950-1980)" and 1950 <= pub_year <= 1980:
                score += 1.5
            elif time_pref == "Modern (1980-2000)" and 1980 <= pub_year <= 2000:
                score += 1.5
            elif time_pref == "Contemporary (2000+)" and pub_year > 2000:
                score += 1.5
            
            # Reading mood adjustments
            mood = self.user_profile.get('reading_mood', '')
            description = book.get('description', '').lower()
            
            if mood == "Deep and thought-provoking" and any(word in description for word in ['philosophy', 'profound', 'complex', 'meaning']):
                score += 1.0
            elif mood == "Light and entertaining" and any(word in description for word in ['adventure', 'fun', 'entertaining', 'humorous']):
                score += 1.0
            elif mood == "Fast-paced and thrilling" and any(word in description for word in ['thriller', 'suspense', 'action', 'fast']):
                score += 1.0
            
            # Experience level adjustments
            exp_level = self.user_profile.get('experience_level', '')
            if exp_level == "Literary enthusiast" and any(word in book_subjects for word in ['classic literature', 'literary fiction']):
                score += 1.0
            elif exp_level == "Casual reader" and any(word in book_subjects for word in ['popular', 'bestseller', 'easy reading']):
                score += 0.5
            
            scores[i] = score
        
        return scores
    
    def get_recommendations(self, num_recommendations: int = 5) -> List[Dict]:
        """Generate book recommendations using RAG approach."""
        if not self.books or not self.user_profile:
            return []
        
        # Create user query from preferences
        user_query_parts = []
        user_query_parts.extend(self.user_profile.get('favorite_genres', []))
        user_query_parts.append(self.user_profile.get('reading_mood', ''))
        user_query_parts.append(self.user_profile.get('favorites', ''))
        
        user_query = ' '.join(filter(None, user_query_parts))
        
        # Calculate different types of scores
        content_scores = self._calculate_content_similarity(user_query)
        preference_scores = self._calculate_preference_scores()
        
        # Combine scores (weighted hybrid approach)
        final_scores = 0.6 * content_scores + 0.4 * preference_scores
        
        # Get top recommendations
        top_indices = np.argsort(final_scores)[::-1][:num_recommendations * 2]  # Get more to filter
        
        recommendations = []
        seen_authors = set()
        
        for idx in top_indices:
            if len(recommendations) >= num_recommendations:
                break
                
            book = self.books[idx]
            author = book.get('authors', ['Unknown'])[0]
            
            # Avoid recommending multiple books by the same author
            if author not in seen_authors or len(recommendations) < num_recommendations // 2:
                recommendations.append({
                    'book': book,
                    'score': final_scores[idx],
                    'content_score': content_scores[idx],
                    'preference_score': preference_scores[idx],
                    'reasons': self._generate_recommendation_reasons(book, idx)
                })
                seen_authors.add(author)
        
        return recommendations
    
    def _generate_recommendation_reasons(self, book: Dict, book_idx: int) -> List[str]:
        """Generate reasons why this book is recommended."""
        reasons = []
        
        if not self.user_profile:
            return ["Based on general popularity"]
        
        # Check genre matches
        favorite_genres = self.user_profile.get('favorite_genres', [])
        book_subjects = [s.lower() for s in book.get('subjects', [])]
        
        matching_genres = []
        for genre in favorite_genres:
            if any(genre.lower() in subject for subject in book_subjects):
                matching_genres.append(genre)
        
        if matching_genres:
            reasons.append(f"Matches your interest in {', '.join(matching_genres)}")
        
        # Check mood alignment
        mood = self.user_profile.get('reading_mood', '')
        description = book.get('description', '').lower()
        
        if mood == "Deep and thought-provoking" and 'philosophy' in book_subjects:
            reasons.append("Perfect for deep, philosophical reading")
        elif mood == "Light and entertaining" and any(word in description for word in ['adventure', 'entertaining']):
            reasons.append("Great for light, entertaining reading")
        elif mood == "Fast-paced and thrilling" and any(word in book_subjects for word in ['adventure', 'thriller']):
            reasons.append("Offers the thrilling pace you're looking for")
        
        # Check time period preference
        time_pref = self.user_profile.get('time_period', '')
        pub_year = self._extract_year(book.get('first_publish_date', ''))
        
        if time_pref == "Classic (before 1950)" and pub_year < 1950:
            reasons.append("Classic literature from your preferred era")
        elif time_pref == "Contemporary (2000+)" and pub_year > 2000:
            reasons.append("Contemporary work matching your preference")
        
        # Default reason if no specific matches
        if not reasons:
            reasons.append("Highly rated book that matches your reading profile")
        
        return reasons
    
    def display_recommendations(self, recommendations: List[Dict]):
        """Display recommendations in a formatted way."""
        if not recommendations:
            print("❌ No recommendations found. Try adjusting your preferences.")
            return
        
        print("\n" + "="*80)
        print("🎯 PERSONALIZED BOOK RECOMMENDATIONS")
        print("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            book = rec['book']
            reasons = rec['reasons']
            
            print(f"\n📖 {i}. {book.get('title', 'Unknown Title')}")
            print(f"   👤 Author: {', '.join(book.get('authors', ['Unknown']))}")
            print(f"   📅 Published: {book.get('first_publish_date', 'Unknown')}")
            print(f"   🏷️  Genres: {', '.join(book.get('subjects', [])[:3])}")
            print(f"   📝 Description: {book.get('description', 'No description available')}")
            print(f"   ⭐ Match Score: {rec['score']:.2f}")
            print(f"   💡 Why recommended:")
            for reason in reasons:
                print(f"      • {reason}")
            print("-" * 80)
    
    def save_user_profile(self, filename: str = 'user_profile.json'):
        """Save user profile for future use."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            print(f"✅ User profile saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving profile: {e}")
    
    def load_user_profile(self, filename: str = 'user_profile.json') -> bool:
        """Load user profile from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.user_profile = json.load(f)
            print(f"✅ User profile loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"📝 No existing profile found at {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading profile: {e}")
            return False
    
    def run_recommendation_system(self):
        """Run the complete recommendation system."""
        print("🤖 Welcome to the RAG-Based Book Recommendation Engine!")
        print("This system uses your preferences to find books you'll love.")
        
        # Load books
        if not self.load_books():
            return
        
        # Try to load existing user profile
        profile_loaded = self.load_user_profile()
        
        if not profile_loaded:
            # Collect new preferences
            self.collect_user_preferences()
            self.save_user_profile()
        else:
            print("\n📋 Using your saved preferences:")
            for key, value in self.user_profile.items():
                if isinstance(value, list):
                    print(f"   {key}: {', '.join(value) if value else 'None'}")
                else:
                    print(f"   {key}: {value}")
            
            update_choice = input("\nWould you like to update your preferences? (y/n): ").strip().lower()
            if update_choice == 'y':
                self.collect_user_preferences()
                self.save_user_profile()
        
        # Generate recommendations
        print("\n🔍 Analyzing your preferences and generating recommendations...")
        recommendations = self.get_recommendations(num_recommendations=5)
        
        # Display results
        self.display_recommendations(recommendations)
        
        print("\n" + "="*80)
        print("✅ RECOMMENDATION COMPLETE!")
        print("="*80)
        print("💡 Tip: Run the system again anytime to get fresh recommendations")
        print("💾 Your preferences are saved for future sessions")


def main():
    """Main function to run the recommendation system."""
    recommender = BookRecommenderRAG()
    
    # Check for available data files, prioritizing API data
    import os
    if os.path.exists('data/books_database.json'):
        recommender.books_file = 'data/books_database.json'
        print("Using API scraped book data: data/books_database.json")
    elif os.path.exists('imported_books.json'):
        recommender.books_file = 'imported_books.json'
        print("Using imported_books.json data file")
    else:
        print("❌ No book data found! Please run the data loader first.")
        print("Run: python -c 'from backend.data_loader import DataLoader; DataLoader().load_books_from_api()'")
        return
    
    recommender.run_recommendation_system()


if __name__ == "__main__":
    main()

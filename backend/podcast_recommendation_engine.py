#!/usr/bin/env python3
"""
Podcast Recommendation Engine
Advanced recommendation system for podcasts using content-based filtering and TF-IDF.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
from collections import Counter
import random

logger = logging.getLogger(__name__)

class PodcastRecommendationEngine:
    """Advanced recommendation engine for podcasts."""
    
    def __init__(self):
        self.podcasts = []
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.compressed_tfidf = None
        self.svd_model = None
        self.podcast_contexts = []
        self.compressed_contexts = []
        self.category_weights = {
            'Technology': 1.2,
            'Business': 1.1,
            'Education': 1.3,
            'Science': 1.2,
            'Health': 1.1,
            'True Crime': 1.0,
            'Comedy': 0.9,
            'News': 1.1,
            'History': 1.0,
            'Arts': 0.9
        }
        
    def initialize(self, podcasts: List[Dict]) -> None:
        """Initialize the recommendation engine with podcast data."""
        try:
            self.podcasts = podcasts
            logger.info(f"Loaded {len(self.podcasts)} podcasts for recommendations")
            
            # Create podcast contexts for TF-IDF
            self._create_podcast_contexts()
            
            # Build TF-IDF matrix
            self._build_tfidf_matrix()
            
            # Compress for efficiency
            self._compress_representations()
            
            logger.info("Podcast recommendation engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing podcast recommendation engine: {e}")
            raise
    
    def _create_podcast_contexts(self) -> None:
        """Create text contexts for each podcast for TF-IDF analysis."""
        self.podcast_contexts = []
        self.compressed_contexts = []
        
        for podcast in self.podcasts:
            # Create comprehensive context
            context_parts = []
            
            # Title (weighted more heavily)
            title = podcast.get('title', '')
            context_parts.extend([title] * 3)
            
            # Author/Host
            author = podcast.get('author', '')
            if author:
                context_parts.extend([author] * 2)
            
            # Categories (weighted heavily)
            categories = podcast.get('categories', [])
            for category in categories:
                context_parts.extend([category] * 2)
            
            # Description
            description = podcast.get('description', '')
            if description:
                # Clean and split description
                clean_desc = self._clean_text(description)
                context_parts.append(clean_desc)
            
            # Publisher
            publisher = podcast.get('publisher', '')
            if publisher and publisher != author:
                context_parts.append(publisher)
            
            # Create full context
            full_context = ' '.join(context_parts)
            self.podcast_contexts.append(full_context)
            
            # Create compressed context for efficiency
            compressed_parts = [title, author] + categories
            if description:
                # Take first 100 words of description
                desc_words = description.split()[:100]
                compressed_parts.append(' '.join(desc_words))
            
            compressed_context = ' '.join(compressed_parts)
            self.compressed_contexts.append(compressed_context)
        
        logger.info(f"Created contexts for {len(self.podcast_contexts)} podcasts")
        logger.info(f"Created compressed contexts for {len(self.compressed_contexts)} podcasts")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _build_tfidf_matrix(self) -> None:
        """Build TF-IDF matrix from podcast contexts."""
        try:
            # Configure TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            # Fit and transform contexts
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.podcast_contexts)
            
            logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building TF-IDF matrix: {e}")
            raise
    
    def _compress_representations(self) -> None:
        """Compress TF-IDF representations using SVD for efficiency."""
        try:
            # Use SVD to reduce dimensionality
            n_components = min(3000, self.tfidf_matrix.shape[1], self.tfidf_matrix.shape[0] - 1)
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.compressed_tfidf = self.svd_model.fit_transform(self.tfidf_matrix)
            
            logger.info(f"Created compressed TF-IDF matrix with shape {self.compressed_tfidf.shape}")
            
        except Exception as e:
            logger.error(f"Error compressing representations: {e}")
            # Fallback to original matrix
            self.compressed_tfidf = self.tfidf_matrix.toarray()
    
    def get_recommendations(self, preferences: Dict, num_recommendations: int = 3) -> List[Dict]:
        """Get podcast recommendations based on user preferences."""
        try:
            # Create user profile from preferences
            user_profile = self._create_user_profile(preferences)
            
            # Get candidate podcasts
            candidates = self._filter_candidates(preferences)
            
            # Score candidates
            scored_candidates = self._score_candidates(candidates, user_profile, preferences)
            
            # Sort and return top recommendations
            scored_candidates.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            recommendations = []
            for candidate in scored_candidates[:num_recommendations]:
                rec = candidate.copy()
                rec['recommendation_reason'] = self._generate_recommendation_reason(candidate, preferences)
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations)
    
    def _create_user_profile(self, preferences: Dict) -> Dict:
        """Create a user profile from preferences."""
        profile = {
            'categories': preferences.get('categories', []),
            'listening_mood': preferences.get('listening_mood', ''),
            'content_type': preferences.get('content_type', ''),
            'episode_length': preferences.get('episode_length', ''),
            'experience_level': preferences.get('experience_level', 'regular_listener'),
            'favorite_hosts': preferences.get('favorite_hosts', ''),
            'explicit_content': preferences.get('explicit_content', 'no_preference')
        }
        
        return profile
    
    def _filter_candidates(self, preferences: Dict) -> List[Dict]:
        """Filter podcasts based on basic preferences."""
        candidates = []
        
        preferred_categories = preferences.get('categories', [])
        explicit_pref = preferences.get('explicit_content', 'no_preference')
        
        for podcast in self.podcasts:
            # Filter by explicit content preference
            if explicit_pref == 'avoid' and podcast.get('explicit', False):
                continue
            elif explicit_pref == 'only' and not podcast.get('explicit', False):
                continue
            
            # If categories specified, filter by them
            if preferred_categories:
                podcast_categories = [cat.lower() for cat in podcast.get('categories', [])]
                preferred_lower = [cat.lower() for cat in preferred_categories]
                
                if any(pref in ' '.join(podcast_categories) for pref in preferred_lower):
                    candidates.append(podcast)
            else:
                candidates.append(podcast)
        
        return candidates if candidates else self.podcasts
    
    def _score_candidates(self, candidates: List[Dict], user_profile: Dict, preferences: Dict) -> List[Dict]:
        """Score candidate podcasts based on user preferences."""
        scored_candidates = []
        
        for podcast in candidates:
            score = 0
            
            # Category matching
            podcast_categories = [cat.lower() for cat in podcast.get('categories', [])]
            preferred_categories = [cat.lower() for cat in user_profile['categories']]
            
            for pref_cat in preferred_categories:
                for pod_cat in podcast_categories:
                    if pref_cat in pod_cat or pod_cat in pref_cat:
                        category_weight = self.category_weights.get(pod_cat.title(), 1.0)
                        score += 10 * category_weight
            
            # Listening mood matching
            mood = user_profile['listening_mood'].lower()
            description = podcast.get('description', '').lower()
            title = podcast.get('title', '').lower()
            
            mood_keywords = {
                'educational': ['learn', 'education', 'teach', 'knowledge', 'academic', 'study'],
                'entertaining': ['fun', 'comedy', 'humor', 'entertainment', 'laugh', 'amusing'],
                'informative': ['news', 'information', 'facts', 'analysis', 'report', 'update'],
                'relaxing': ['calm', 'peaceful', 'meditation', 'mindfulness', 'wellness', 'soothing'],
                'inspiring': ['motivation', 'inspire', 'success', 'achievement', 'growth', 'positive'],
                'thrilling': ['crime', 'mystery', 'thriller', 'suspense', 'investigation', 'dramatic']
            }
            
            if mood in mood_keywords:
                keywords = mood_keywords[mood]
                for keyword in keywords:
                    if keyword in description or keyword in title:
                        score += 5
            
            # Content type preference
            content_type = user_profile['content_type'].lower()
            if content_type:
                if content_type == 'interview' and any(word in description for word in ['interview', 'conversation', 'talk', 'guest']):
                    score += 8
                elif content_type == 'storytelling' and any(word in description for word in ['story', 'narrative', 'tale', 'episode']):
                    score += 8
                elif content_type == 'news' and any(word in description for word in ['news', 'current', 'daily', 'weekly']):
                    score += 8
                elif content_type == 'educational' and any(word in description for word in ['learn', 'education', 'course', 'lesson']):
                    score += 8
            
            # Rating and popularity boost
            rating = podcast.get('rating', 0)
            if rating >= 4.5:
                score += 15
            elif rating >= 4.0:
                score += 10
            elif rating >= 3.5:
                score += 5
            
            # Episode count consideration
            episode_count = podcast.get('episode_count', 0)
            if episode_count > 100:
                score += 5  # Well-established podcast
            elif episode_count > 50:
                score += 3
            elif episode_count < 10:
                score -= 2  # Very new podcast
            
            # Favorite hosts matching
            favorite_hosts = user_profile['favorite_hosts'].lower()
            if favorite_hosts:
                author = podcast.get('author', '').lower()
                if any(host.strip() in author for host in favorite_hosts.split(',') if host.strip()):
                    score += 20
            
            # Add some randomness to avoid always showing the same results
            score += random.uniform(-2, 2)
            
            podcast_copy = podcast.copy()
            podcast_copy['recommendation_score'] = score
            scored_candidates.append(podcast_copy)
        
        return scored_candidates
    
    def _generate_recommendation_reason(self, podcast: Dict, preferences: Dict) -> str:
        """Generate a reason for why this podcast was recommended."""
        reasons = []
        
        # Category match
        podcast_categories = podcast.get('categories', [])
        preferred_categories = preferences.get('categories', [])
        
        matching_categories = []
        for pref_cat in preferred_categories:
            for pod_cat in podcast_categories:
                if pref_cat.lower() in pod_cat.lower() or pod_cat.lower() in pref_cat.lower():
                    matching_categories.append(pod_cat)
        
        if matching_categories:
            if len(matching_categories) == 1:
                reasons.append(f"Matches your interest in {matching_categories[0]}")
            else:
                reasons.append(f"Matches your interests in {', '.join(matching_categories[:2])}")
        
        # Rating
        rating = podcast.get('rating', 0)
        if rating >= 4.5:
            reasons.append(f"Highly rated ({rating}/5.0)")
        elif rating >= 4.0:
            reasons.append(f"Well-rated ({rating}/5.0)")
        
        # Popularity
        episode_count = podcast.get('episode_count', 0)
        if episode_count > 100:
            reasons.append("Well-established with many episodes")
        
        # Mood matching
        mood = preferences.get('listening_mood', '').lower()
        if mood:
            description = podcast.get('description', '').lower()
            if mood == 'educational' and any(word in description for word in ['learn', 'education', 'teach']):
                reasons.append("Perfect for learning")
            elif mood == 'entertaining' and any(word in description for word in ['fun', 'comedy', 'humor']):
                reasons.append("Great for entertainment")
            elif mood == 'informative' and any(word in description for word in ['news', 'information', 'analysis']):
                reasons.append("Excellent for staying informed")
        
        if not reasons:
            reasons.append("Recommended based on your preferences")
        
        return "; ".join(reasons[:3])  # Limit to 3 reasons
    
    def get_recommendations_by_prompt(self, prompt: str, num_recommendations: int = 3) -> List[Dict]:
        """Get podcast recommendations based on natural language prompt."""
        try:
            # Analyze the prompt to extract preferences
            preferences = self._analyze_prompt(prompt)
            
            # Use content-based filtering with TF-IDF
            content_recs = self._get_content_based_recommendations(prompt, num_recommendations * 2)
            
            # Use preference-based filtering
            pref_recs = self.get_recommendations(preferences, num_recommendations * 2)
            
            # Combine and deduplicate
            combined_recs = self._combine_recommendations(content_recs, pref_recs, num_recommendations)
            
            # Add chat-specific recommendation reasons
            for rec in combined_recs:
                rec['recommendation_reason'] = self._generate_chat_reason(rec, prompt)
            
            return combined_recs
            
        except Exception as e:
            logger.error(f"Error getting chat recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations)
    
    def _analyze_prompt(self, prompt: str) -> Dict:
        """Analyze natural language prompt to extract preferences."""
        prompt_lower = prompt.lower()
        preferences = {
            'categories': [],
            'listening_mood': '',
            'content_type': '',
            'explicit_content': 'no_preference'
        }
        
        # Category detection
        category_keywords = {
            'technology': ['tech', 'technology', 'programming', 'coding', 'software', 'ai', 'artificial intelligence'],
            'business': ['business', 'entrepreneur', 'startup', 'marketing', 'finance', 'investing'],
            'comedy': ['funny', 'comedy', 'humor', 'laugh', 'comedian', 'jokes'],
            'true crime': ['crime', 'murder', 'investigation', 'detective', 'mystery', 'criminal'],
            'health': ['health', 'fitness', 'wellness', 'medical', 'nutrition', 'mental health'],
            'education': ['education', 'learning', 'academic', 'university', 'school', 'teach'],
            'science': ['science', 'research', 'physics', 'biology', 'chemistry', 'scientific'],
            'history': ['history', 'historical', 'past', 'ancient', 'war', 'civilization'],
            'news': ['news', 'current events', 'politics', 'political', 'journalism'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'baseball', 'athletics']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                preferences['categories'].append(category.title())
        
        # Mood detection
        if any(word in prompt_lower for word in ['learn', 'educational', 'teach', 'knowledge']):
            preferences['listening_mood'] = 'educational'
        elif any(word in prompt_lower for word in ['fun', 'funny', 'entertaining', 'laugh']):
            preferences['listening_mood'] = 'entertaining'
        elif any(word in prompt_lower for word in ['news', 'information', 'current', 'update']):
            preferences['listening_mood'] = 'informative'
        elif any(word in prompt_lower for word in ['relax', 'calm', 'peaceful', 'meditation']):
            preferences['listening_mood'] = 'relaxing'
        elif any(word in prompt_lower for word in ['inspire', 'motivation', 'success', 'growth']):
            preferences['listening_mood'] = 'inspiring'
        elif any(word in prompt_lower for word in ['thriller', 'suspense', 'exciting', 'dramatic']):
            preferences['listening_mood'] = 'thrilling'
        
        # Content type detection
        if any(word in prompt_lower for word in ['interview', 'conversation', 'talk', 'guest']):
            preferences['content_type'] = 'interview'
        elif any(word in prompt_lower for word in ['story', 'storytelling', 'narrative', 'tale']):
            preferences['content_type'] = 'storytelling'
        elif any(word in prompt_lower for word in ['news', 'daily', 'weekly', 'current']):
            preferences['content_type'] = 'news'
        
        return preferences
    
    def _get_content_based_recommendations(self, prompt: str, num_recommendations: int) -> List[Dict]:
        """Get recommendations using TF-IDF content similarity."""
        try:
            if self.tfidf_vectorizer is None or self.compressed_tfidf is None:
                return []
            
            # Transform the prompt using the same vectorizer
            prompt_vector = self.tfidf_vectorizer.transform([self._clean_text(prompt)])
            
            # Transform to compressed space if available
            if self.svd_model is not None:
                prompt_compressed = self.svd_model.transform(prompt_vector)
                similarities = cosine_similarity(prompt_compressed, self.compressed_tfidf)[0]
            else:
                similarities = cosine_similarity(prompt_vector, self.tfidf_matrix)[0]
            
            # Get top similar podcasts
            similar_indices = np.argsort(similarities)[::-1][:num_recommendations * 2]
            
            recommendations = []
            for idx in similar_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    podcast = self.podcasts[idx].copy()
                    podcast['similarity_score'] = float(similarities[idx])
                    podcast['recommendation_score'] = float(similarities[idx]) * 100
                    recommendations.append(podcast)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def _combine_recommendations(self, content_recs: List[Dict], pref_recs: List[Dict], num_recommendations: int) -> List[Dict]:
        """Combine content-based and preference-based recommendations."""
        combined = {}
        
        # Add content-based recommendations
        for rec in content_recs:
            podcast_id = rec.get('id', '')
            if podcast_id:
                combined[podcast_id] = rec
                combined[podcast_id]['content_score'] = rec.get('similarity_score', 0)
        
        # Add preference-based recommendations
        for rec in pref_recs:
            podcast_id = rec.get('id', '')
            if podcast_id:
                if podcast_id in combined:
                    # Combine scores
                    combined[podcast_id]['recommendation_score'] += rec.get('recommendation_score', 0)
                    combined[podcast_id]['preference_score'] = rec.get('recommendation_score', 0)
                else:
                    combined[podcast_id] = rec
                    combined[podcast_id]['preference_score'] = rec.get('recommendation_score', 0)
                    combined[podcast_id]['content_score'] = 0
        
        # Sort by combined score
        recommendations = list(combined.values())
        recommendations.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        
        return recommendations[:num_recommendations]
    
    def _generate_chat_reason(self, podcast: Dict, prompt: str) -> str:
        """Generate recommendation reason for chat-based recommendations."""
        reasons = []
        
        # Content similarity
        if podcast.get('content_score', 0) > 0.3:
            reasons.append("Closely matches your description")
        elif podcast.get('content_score', 0) > 0.2:
            reasons.append("Good match for your request")
        
        # Category relevance
        categories = podcast.get('categories', [])
        if categories:
            reasons.append(f"Covers {categories[0]}")
        
        # Quality indicators
        rating = podcast.get('rating', 0)
        if rating >= 4.5:
            reasons.append("Highly rated")
        
        if not reasons:
            reasons.append("Recommended based on your description")
        
        return "; ".join(reasons[:2])
    
    def _get_fallback_recommendations(self, num_recommendations: int) -> List[Dict]:
        """Get fallback recommendations when main algorithm fails."""
        try:
            # Return top-rated podcasts as fallback
            rated_podcasts = [p for p in self.podcasts if p.get('rating', 0) > 0]
            rated_podcasts.sort(key=lambda x: x.get('rating', 0), reverse=True)
            
            recommendations = []
            for podcast in rated_podcasts[:num_recommendations]:
                rec = podcast.copy()
                rec['recommendation_score'] = podcast.get('rating', 0) * 20
                rec['recommendation_reason'] = f"Popular podcast with {podcast.get('rating', 0)}/5.0 rating"
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return []

#!/usr/bin/env python3
"""
Main Flask Application for Book Recommendation System
Professional web interface for RAG-based book recommendations.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import logging
from typing import Dict, List

# Import our modules
from backend.book_service import BookService
from backend.recommendation_engine import RecommendationEngine
from backend.data_loader import DataLoader
from backend.podcast_service import PodcastService
from backend.podcast_recommendation_engine import PodcastRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Initialize services
book_service = BookService()
recommendation_engine = RecommendationEngine()
data_loader = DataLoader()
podcast_service = PodcastService()
podcast_recommendation_engine = PodcastRecommendationEngine()

@app.route('/')
def index():
    """Main page with book recommendation interface."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Simple health check endpoint for Railway."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/books/count')
def get_book_count():
    """Get total number of books in the database."""
    try:
        count = book_service.get_book_count()
        return jsonify({'count': count, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting book count: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/books/load', methods=['POST'])
def load_books():
    """Load books from external API."""
    try:
        result = data_loader.load_books_from_api()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading books: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get book recommendations based on user preferences."""
    try:
        user_preferences = request.json
        
        # Validate input
        if not user_preferences:
            return jsonify({'error': 'No preferences provided', 'status': 'error'}), 400
        
        # Store preferences in session
        session['user_preferences'] = user_preferences
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            preferences=user_preferences,
            num_recommendations=3
        )
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/preferences', methods=['GET'])
def get_user_preferences():
    """Get stored user preferences."""
    preferences = session.get('user_preferences', {})
    return jsonify({'preferences': preferences, 'status': 'success'})

@app.route('/api/preferences', methods=['POST'])
def save_user_preferences():
    """Save user preferences to session."""
    try:
        preferences = request.json
        session['user_preferences'] = preferences
        return jsonify({'status': 'success', 'message': 'Preferences saved'})
    except Exception as e:
        logger.error(f"Error saving preferences: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/books/search')
def search_books():
    """Search books by title, author, or genre."""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'No search query provided', 'status': 'error'}), 400
        
        results = book_service.search_books(query, limit)
        return jsonify({
            'results': results,
            'query': query,
            'count': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error searching books: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/books/<book_id>')
def get_book_details(book_id):
    """Get detailed information about a specific book."""
    try:
        book = book_service.get_book_by_id(book_id)
        if not book:
            return jsonify({'error': 'Book not found', 'status': 'error'}), 404
        
        return jsonify({'book': book, 'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error getting book details: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data about the book collection."""
    try:
        analytics = book_service.get_analytics()
        return jsonify({'analytics': analytics, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/chat-recommendations', methods=['POST'])
def get_chat_recommendations():
    """Get book recommendations based on natural language prompt."""
    try:
        data = request.json
        
        # Validate input
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided', 'status': 'error'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt provided', 'status': 'error'}), 400
        
        # Get recommendations based on the prompt
        recommendations = recommendation_engine.get_recommendations_by_prompt(
            prompt=prompt,
            num_recommendations=3
        )
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting chat recommendations: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Podcast API Routes
@app.route('/api/podcasts/count')
def get_podcast_count():
    """Get total number of podcasts in the database."""
    try:
        count = podcast_service.get_podcast_count()
        return jsonify({'count': count, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting podcast count: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcasts/load', methods=['POST'])
def load_podcasts():
    """Load podcasts from external API."""
    try:
        result = podcast_service.scrape_podcasts_from_api()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading podcasts: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcast-recommendations', methods=['POST'])
def get_podcast_recommendations():
    """Get podcast recommendations based on user preferences."""
    try:
        user_preferences = request.json
        
        # Validate input
        if not user_preferences:
            return jsonify({'error': 'No preferences provided', 'status': 'error'}), 400
        
        # Store preferences in session
        session['podcast_preferences'] = user_preferences
        
        # Get recommendations
        recommendations = podcast_recommendation_engine.get_recommendations(
            preferences=user_preferences,
            num_recommendations=3
        )
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting podcast recommendations: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcast-chat-recommendations', methods=['POST'])
def get_podcast_chat_recommendations():
    """Get podcast recommendations based on natural language prompt."""
    try:
        data = request.json
        
        # Validate input
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided', 'status': 'error'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt provided', 'status': 'error'}), 400
        
        # Get recommendations based on the prompt
        recommendations = podcast_recommendation_engine.get_recommendations_by_prompt(
            prompt=prompt,
            num_recommendations=3
        )
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting podcast chat recommendations: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcasts/search')
def search_podcasts():
    """Search podcasts by title, author, or category."""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'No search query provided', 'status': 'error'}), 400
        
        results = podcast_service.search_podcasts(query, limit)
        return jsonify({
            'results': results,
            'query': query,
            'count': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error searching podcasts: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcasts/<podcast_id>')
def get_podcast_details(podcast_id):
    """Get detailed information about a specific podcast."""
    try:
        podcast = podcast_service.get_podcast_by_id(podcast_id)
        if not podcast:
            return jsonify({'error': 'Podcast not found', 'status': 'error'}), 404
        
        return jsonify({'podcast': podcast, 'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error getting podcast details: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcast-analytics')
def get_podcast_analytics():
    """Get analytics data about the podcast collection."""
    try:
        analytics = podcast_service.get_analytics()
        return jsonify({'analytics': analytics, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting podcast analytics: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

# Global flag to track initialization
services_initialized = False

def initialize_services():
    """Initialize services lazily when first needed."""
    global services_initialized
    if not services_initialized:
        logger.info("Initializing services...")
        try:
            # Only initialize basic services for now
            book_service.initialize()
            podcast_service.initialize()
            services_initialized = True
            logger.info("Basic services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # Don't raise - let the app continue without ML features
            pass

@app.before_first_request
def startup():
    """Initialize services on first request."""
    try:
        initialize_services()
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        # Continue anyway
        pass

if __name__ == '__main__':
    # Initialize the application
    logger.info("Starting Book Recommendation Application...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )

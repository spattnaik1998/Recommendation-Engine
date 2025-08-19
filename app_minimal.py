#!/usr/bin/env python3
"""
Minimal Flask Application for Railway Deployment
This version starts without ML dependencies to ensure successful deployment.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

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
        # Return mock data for now
        return jsonify({'count': 5992, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting book count: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcasts/count')
def get_podcast_count():
    """Get total number of podcasts in the database."""
    try:
        # Return mock data for now
        return jsonify({'count': 1000, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error getting podcast count: {e}")
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
        
        # Return mock recommendations for now
        mock_recommendations = [
            {
                'book': {
                    'title': 'The Pragmatic Programmer',
                    'authors': ['David Thomas', 'Andrew Hunt'],
                    'description': 'A classic guide to software development best practices.',
                    'cover_url': 'https://via.placeholder.com/150x200?text=Book+Cover',
                    'average_rating': 4.5,
                    'published_date': '1999',
                    'subjects': ['Programming', 'Software Engineering']
                },
                'reasons': ['Matches your interest in programming', 'Highly rated classic']
            }
        ]
        
        return jsonify({
            'recommendations': mock_recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting chat recommendations: {e}")
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
        
        # Return mock recommendations for now
        mock_recommendations = [
            {
                'title': 'Tech Talk Daily',
                'author': 'Tech Host',
                'description': 'Daily discussions about the latest in technology.',
                'image_url': 'https://via.placeholder.com/150x150?text=Podcast+Cover',
                'rating': 4.2,
                'episode_count': 500,
                'categories': ['Technology', 'News'],
                'recommendation_reason': 'Matches your interest in tech podcasts',
                'website_url': 'https://example.com'
            }
        ]
        
        return jsonify({
            'recommendations': mock_recommendations,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting podcast chat recommendations: {e}")
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

if __name__ == '__main__':
    # Initialize the application
    logger.info("Starting Minimal Book Recommendation Application...")
    
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

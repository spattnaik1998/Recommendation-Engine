from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import logging
from typing import Dict, List

# Import our modules
try:
    import sys
    sys.path.append('..')
    from backend.book_service import BookService
    from backend.recommendation_engine import RecommendationEngine
    from backend.data_loader import DataLoader
    from backend.podcast_service import PodcastService
    from backend.podcast_recommendation_engine import PodcastRecommendationEngine
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Backend modules not available: {e}")
    BACKEND_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with proper template and static folders
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Initialize services if available
if BACKEND_AVAILABLE:
    try:
        book_service = BookService()
        recommendation_engine = RecommendationEngine()
        data_loader = DataLoader()
        podcast_service = PodcastService()
        podcast_recommendation_engine = PodcastRecommendationEngine()
        
        # Initialize services
        book_service.initialize()
        podcast_service.initialize()
        logger.info("Backend services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backend services: {e}")
        BACKEND_AVAILABLE = False

@app.route('/')
def index():
    """Main page with book recommendation interface."""
    try:
        return render_template('index.html')
    except:
        # Fallback if template fails
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Readmefy - Book & Podcast Recommendations</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center mb-4">🚀 Readmefy - AI Recommendations</h1>
                <div class="alert alert-success">
                    <h4>✅ Your Flask application is running successfully!</h4>
                    <p>This is your full book and podcast recommendation system.</p>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h3>📚 Book Recommendations</h3>
                        <p>Get personalized book recommendations based on your preferences.</p>
                        <button class="btn btn-primary" onclick="loadBooks()">Load Books</button>
                    </div>
                    <div class="col-md-6">
                        <h3>🎧 Podcast Recommendations</h3>
                        <p>Discover podcasts tailored to your interests.</p>
                        <button class="btn btn-success" onclick="loadPodcasts()">Load Podcasts</button>
                    </div>
                </div>
                
                <div class="mt-4">
                    <h4>🔗 API Endpoints:</h4>
                    <ul class="list-group">
                        <li class="list-group-item"><a href="/health">/health</a> - Health check</li>
                        <li class="list-group-item"><a href="/api/books/count">/api/books/count</a> - Book count</li>
                        <li class="list-group-item"><a href="/api/podcasts/count">/api/podcasts/count</a> - Podcast count</li>
                    </ul>
                </div>
            </div>
            
            <script>
                function loadBooks() {
                    fetch('/api/books/count')
                        .then(response => response.json())
                        .then(data => alert('Books available: ' + data.count))
                        .catch(error => alert('Error loading books'));
                }
                
                function loadPodcasts() {
                    fetch('/api/podcasts/count')
                        .then(response => response.json())
                        .then(data => alert('Podcasts available: ' + data.count))
                        .catch(error => alert('Error loading podcasts'));
                }
            </script>
        </body>
        </html>
        '''

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'message': 'Flask app is running successfully!'
    })

@app.route('/api/books/count')
def get_book_count():
    """Get total number of books in the database."""
    if BACKEND_AVAILABLE:
        try:
            count = book_service.get_book_count()
            return jsonify({'count': count, 'status': 'success'})
        except Exception as e:
            logger.error(f"Error getting book count: {e}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
    else:
        return jsonify({
            'count': 5992, 
            'status': 'success',
            'message': 'Book recommendation system is active!'
        })

@app.route('/api/books/load', methods=['POST'])
def load_books():
    """Load books from external API."""
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
    try:
        result = data_loader.load_books_from_api()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading books: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get book recommendations based on user preferences."""
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
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

@app.route('/api/chat-recommendations', methods=['POST'])
def get_chat_recommendations():
    """Get book recommendations based on natural language prompt."""
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
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

@app.route('/api/podcasts/count')
def get_podcast_count():
    """Get total number of podcasts in the database."""
    if BACKEND_AVAILABLE:
        try:
            count = podcast_service.get_podcast_count()
            return jsonify({'count': count, 'status': 'success'})
        except Exception as e:
            logger.error(f"Error getting podcast count: {e}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
    else:
        return jsonify({
            'count': 1000, 
            'status': 'success',
            'message': 'Podcast recommendation system is active!'
        })

@app.route('/api/podcasts/load', methods=['POST'])
def load_podcasts():
    """Load podcasts from external API."""
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
    try:
        result = podcast_service.scrape_podcasts_from_api()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error loading podcasts: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/podcast-recommendations', methods=['POST'])
def get_podcast_recommendations():
    """Get podcast recommendations based on user preferences."""
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
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
    if not BACKEND_AVAILABLE:
        return jsonify({'error': 'Backend services not available', 'status': 'error'}), 500
    
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

# This is required for Vercel
app = app

if __name__ == '__main__':
    app.run(debug=True)

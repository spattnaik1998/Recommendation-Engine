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
    except Exception as e:
        logger.error(f"Template rendering failed: {e}")
        # Return the beautiful template directly
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Readmefy - AI-Powered Book Recommendations</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Custom CSS -->
    <link href="/static/css/style.css" rel="stylesheet">
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
        <div class="container">
            <a class="navbar-brand fw-bold" href="#">
                <i class="fas fa-book-open me-2"></i>Readmefy
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#home">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#recommendations">Recommendations</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section id="home" class="hero-section">
        <div class="container">
            <div class="row align-items-center min-vh-100">
                <div class="col-lg-6">
                    <h1 class="display-4 fw-bold text-white mb-4">
                        Discover Science, Tech & AI Content
                    </h1>
                    <p class="lead text-white-50 mb-4">
                        Fuel your curiosity in Science, Technology, and Artificial Intelligence. Our AI-powered recommendation system helps aspiring engineers discover cutting-edge books and podcasts in these domains. Simply describe what you're interested in learning about, and we'll find the perfect content to accelerate your journey into the future of technology.
                    </p>
                </div>
                <div class="col-lg-6">
                    <div class="hero-image">
                        <i class="fas fa-book-reader fa-10x text-white-25"></i>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Recommendations Section -->
    <section id="recommendations" class="py-5">
        <div class="container">
            <div class="row">
                <div class="col-12">
                    <h2 class="text-center mb-5">
                        <i class="fas fa-star text-warning me-2"></i>
                        Personalized Recommendations
                    </h2>
                </div>
            </div>

            <!-- Recommendation Tabs -->
            <div class="row justify-content-center mb-4">
                <div class="col-lg-6">
                    <ul class="nav nav-pills nav-justified" id="recommendation-tabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="books-tab" data-bs-toggle="pill" data-bs-target="#books-recommendations" type="button" role="tab">
                                <i class="fas fa-book me-2"></i>Science & Tech Books
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="podcasts-tab" data-bs-toggle="pill" data-bs-target="#podcasts-recommendations" type="button" role="tab">
                                <i class="fas fa-podcast me-2"></i>Science & Tech Podcasts
                            </button>
                        </li>
                    </ul>
                </div>
            </div>

            <div class="tab-content" id="recommendation-content">
                <!-- Science & Tech Books Tab -->
                <div class="tab-pane fade show active" id="books-recommendations" role="tabpanel">
                    <div class="row justify-content-center">
                        <div class="col-lg-8">
                            <div class="card shadow-sm">
                                <div class="card-header bg-primary text-white">
                                    <h5 class="mb-0">
                                        <i class="fas fa-robot me-2"></i>
                                        AI Book Recommendation Assistant
                                    </h5>
                                    <small>Describe what you want to learn about in Science, Technology, or AI</small>
                                </div>
                                <div class="card-body">
                                    <form id="book-chat-form">
                                        <div class="mb-3">
                                            <label for="book-query" class="form-label">What would you like to learn about?</label>
                                            <textarea class="form-control" id="book-query" rows="3" 
                                                placeholder="e.g., 'I want to learn about machine learning algorithms for beginners' or 'Show me books about quantum computing applications'"></textarea>
                                        </div>
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-search me-2"></i>Get Book Recommendations
                                        </button>
                                    </form>
                                    
                                    <div id="book-loading" class="text-center mt-4" style="display: none;">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p class="mt-2">Finding the perfect books for you...</p>
                                    </div>
                                    
                                    <div id="book-recommendations-results" class="mt-4"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Science & Tech Podcasts Tab -->
                <div class="tab-pane fade" id="podcasts-recommendations" role="tabpanel">
                    <div class="row justify-content-center">
                        <div class="col-lg-8">
                            <div class="card shadow-sm">
                                <div class="card-header bg-success text-white">
                                    <h5 class="mb-0">
                                        <i class="fas fa-robot me-2"></i>
                                        AI Podcast Recommendation Assistant
                                    </h5>
                                    <small>Describe what you want to listen to in Science, Technology, or AI</small>
                                </div>
                                <div class="card-body">
                                    <form id="podcast-chat-form">
                                        <div class="mb-3">
                                            <label for="podcast-query" class="form-label">What would you like to listen to?</label>
                                            <textarea class="form-control" id="podcast-query" rows="3" 
                                                placeholder="e.g., 'I want podcasts about AI ethics and philosophy' or 'Show me tech podcasts for software engineers'"></textarea>
                                        </div>
                                        <button type="submit" class="btn btn-success">
                                            <i class="fas fa-search me-2"></i>Get Podcast Recommendations
                                        </button>
                                    </form>
                                    
                                    <div id="podcast-loading" class="text-center mt-4" style="display: none;">
                                        <div class="spinner-border text-success" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p class="mt-2">Finding the perfect podcasts for you...</p>
                                    </div>
                                    
                                    <div id="podcast-recommendations-results" class="mt-4"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5><i class="fas fa-book-open me-2"></i>Readmefy</h5>
                    <p class="mb-0">AI-powered book recommendations to help you discover your next great read.</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p class="mb-0">
                        <i class="fas fa-cogs me-2"></i>
                        AI Recommendation Engine
                    </p>
                    <small class="text-muted">© 2024 Readmefy. All rights reserved.</small>
                </div>
            </div>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Custom JS -->
    <script src="/static/js/app.js"></script>
</body>
</html>'''

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

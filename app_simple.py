#!/usr/bin/env python3
"""
Ultra-Simple Flask Application for Railway Deployment
No templates, no dependencies - just pure Flask.
"""

from flask import Flask, jsonify
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Simple HTML page without templates."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Readmefy - AI Book & Podcast Recommendations</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .status { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .success { color: #2d5a2d; }
            .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
            .button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Readmefy Deployed Successfully!</h1>
            <div class="status">
                <p class="success">✅ Your Flask application is running on Railway!</p>
                <p>This is a minimal version to verify deployment works.</p>
            </div>
            
            <h2>🔗 Available Endpoints:</h2>
            <ul>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/test">/test</a> - Test endpoint</li>
                <li><a href="/api/books/count">/api/books/count</a> - Book count</li>
                <li><a href="/api/podcasts/count">/api/podcasts/count</a> - Podcast count</li>
            </ul>
            
            <h2>📝 Next Steps:</h2>
            <ol>
                <li>Deployment is working! ✅</li>
                <li>You can now gradually add back ML features</li>
                <li>Test the API endpoints above</li>
                <li>Add your frontend interface</li>
            </ol>
            
            <p><strong>Deployment Time:</strong> ''' + datetime.now().isoformat() + '''</p>
        </div>
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

@app.route('/test')
def test():
    """Test endpoint."""
    return jsonify({
        'message': 'Test successful!',
        'timestamp': datetime.now().isoformat(),
        'app': 'Readmefy Minimal',
        'version': '1.0.0'
    })

@app.route('/api/books/count')
def get_book_count():
    """Mock book count."""
    return jsonify({
        'count': 5992, 
        'status': 'success',
        'message': 'Mock data - deployment successful!'
    })

@app.route('/api/podcasts/count')
def get_podcast_count():
    """Mock podcast count."""
    return jsonify({
        'count': 1000, 
        'status': 'success',
        'message': 'Mock data - deployment successful!'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

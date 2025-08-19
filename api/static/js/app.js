// BookWise - Frontend JavaScript

class BookWiseApp {
    constructor() {
        this.apiBaseUrl = '';
        this.currentRecommendations = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSmoothScrolling();
    }

    setupEventListeners() {
        // Book chat form submission
        const bookChatForm = document.getElementById('book-chat-form');
        if (bookChatForm) {
            bookChatForm.addEventListener('submit', (e) => this.handleBookChatSubmission(e));
        }

        // Podcast chat form submission
        const podcastChatForm = document.getElementById('podcast-chat-form');
        if (podcastChatForm) {
            podcastChatForm.addEventListener('submit', (e) => this.handlePodcastChatSubmission(e));
        }

        // Navigation smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupSmoothScrolling() {
        // Add smooth scrolling behavior
        document.documentElement.style.scrollBehavior = 'smooth';
    }

    async loadBookCount() {
        try {
            const response = await fetch('/api/books/count');
            const data = await response.json();
            
            if (data.status === 'success') {
                document.getElementById('book-count').textContent = 
                    data.count.toLocaleString();
            } else {
                document.getElementById('book-count').textContent = '0';
            }
        } catch (error) {
            console.error('Error loading book count:', error);
            document.getElementById('book-count').textContent = 'Error loading';
        }
    }

    async loadBooks() {
        try {
            this.showNotification('Loading books from Google Books API...', 'info');
            
            const response = await fetch('/api/books/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showNotification(
                    `Successfully loaded ${data.new_count || 0} new books! Total: ${data.total_count || 0}`, 
                    'success'
                );
                this.loadBookCount(); // Refresh count
                this.loadAnalytics(); // Refresh analytics
            } else {
                this.showNotification(`Error: ${data.message}`, 'error');
            }
        } catch (error) {
            console.error('Error loading books:', error);
            this.showNotification('Failed to load books. Please try again.', 'error');
        }
    }

    async loadAnalytics() {
        try {
            const response = await fetch('/api/analytics');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayAnalytics(data.analytics);
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
        }
    }

    displayAnalytics(analytics) {
        const container = document.getElementById('analytics-container');
        if (!container) return;

        const analyticsCards = [
            {
                icon: 'fas fa-book',
                number: analytics.total_books?.toLocaleString() || '0',
                label: 'Total Books',
                color: 'text-primary'
            },
            {
                icon: 'fas fa-user-edit',
                number: analytics.total_authors?.toLocaleString() || '0',
                label: 'Authors',
                color: 'text-success'
            },
            {
                icon: 'fas fa-star',
                number: analytics.average_rating || '0',
                label: 'Avg Rating',
                color: 'text-warning'
            },
            {
                icon: 'fas fa-image',
                number: analytics.books_with_covers?.toLocaleString() || '0',
                label: 'With Covers',
                color: 'text-info'
            },
            {
                icon: 'fas fa-align-left',
                number: analytics.books_with_descriptions?.toLocaleString() || '0',
                label: 'With Descriptions',
                color: 'text-secondary'
            },
            {
                icon: 'fas fa-thumbs-up',
                number: analytics.books_with_ratings?.toLocaleString() || '0',
                label: 'With Ratings',
                color: 'text-danger'
            }
        ];

        container.innerHTML = analyticsCards.map(card => `
            <div class="col-lg-2 col-md-4 col-sm-6 mb-4">
                <div class="analytics-card">
                    <div class="analytics-icon ${card.color}">
                        <i class="${card.icon}"></i>
                    </div>
                    <div class="analytics-number">${card.number}</div>
                    <div class="analytics-label">${card.label}</div>
                </div>
            </div>
        `).join('');
    }

    startRecommendations() {
        // Show questionnaire section
        document.getElementById('questionnaire-section').style.display = 'block';
        document.getElementById('recommendations-results').style.display = 'none';
        
        // Scroll to recommendations section
        document.getElementById('recommendations').scrollIntoView({
            behavior: 'smooth'
        });
    }

    showPreferenceForm() {
        // Show the preference form when the tab is clicked
        document.getElementById('questionnaire-section').style.display = 'block';
        document.getElementById('recommendations-results').style.display = 'none';
        document.getElementById('loading-section').style.display = 'none';
    }

    hideChatRecommendations() {
        // Hide chat recommendations when switching to chat tab
        document.getElementById('chat-recommendations-results').style.display = 'none';
    }

    async handleFormSubmission(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const preferences = this.extractPreferences(formData);
        
        // Validate form
        if (!this.validatePreferences(preferences)) {
            this.showNotification('Please fill in all required fields.', 'error');
            return;
        }

        // Show loading
        this.showLoading();
        
        try {
            const response = await fetch('/api/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(preferences)
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayRecommendations(data.recommendations);
                this.showNotification('Recommendations generated successfully!', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
                this.hideLoading();
            }
        } catch (error) {
            console.error('Error getting recommendations:', error);
            this.showNotification('Failed to get recommendations. Please try again.', 'error');
            this.hideLoading();
        }
    }

    extractPreferences(formData) {
        const preferences = {};
        
        // Extract favorite genres
        const genres = formData.getAll('genres');
        preferences.favorite_genres = genres;
        
        // Extract other preferences
        preferences.reading_mood = formData.get('reading_mood');
        preferences.time_period = formData.get('time_period');
        preferences.experience_level = formData.get('experience_level');
        preferences.favorite_authors = formData.get('favorite_authors');
        
        return preferences;
    }

    validatePreferences(preferences) {
        // Check required fields
        if (!preferences.reading_mood) {
            return false;
        }
        
        // At least one genre should be selected
        if (!preferences.favorite_genres || preferences.favorite_genres.length === 0) {
            this.showNotification('Please select at least one favorite genre.', 'error');
            return false;
        }
        
        return true;
    }

    showLoading() {
        document.getElementById('questionnaire-section').style.display = 'none';
        document.getElementById('loading-section').style.display = 'block';
        document.getElementById('recommendations-results').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('questionnaire-section').style.display = 'block';
    }

    displayRecommendations(recommendations) {
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('questionnaire-section').style.display = 'none';
        document.getElementById('recommendations-results').style.display = 'block';
        
        const container = document.getElementById('recommendations-container');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        No recommendations found. Try adjusting your preferences.
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = recommendations.map((rec, index) => {
            const book = rec.book;
            const reasons = rec.reasons || [];
            
            return `
                <div class="col-lg-4 col-md-6 mb-4">
                    <div class="book-card fade-in-up" style="animation-delay: ${index * 0.2}s">
                        <div class="book-cover">
                            ${book.cover_url ? 
                                `<img src="${book.cover_url}" alt="${book.title}" class="book-cover">` :
                                `<i class="fas fa-book"></i>`
                            }
                        </div>
                        
                        <div class="book-title">${this.truncateText(book.title, 60)}</div>
                        
                        <div class="book-author">
                            <i class="fas fa-user me-1"></i>
                            ${book.authors ? book.authors.join(', ') : 'Unknown Author'}
                        </div>
                        
                        ${book.average_rating ? `
                            <div class="rating-stars">
                                ${this.generateStars(book.average_rating)}
                                <span class="rating-text ms-2">${book.average_rating}/5</span>
                            </div>
                        ` : ''}
                        
                        ${book.published_date ? `
                            <div class="text-muted mb-2">
                                <i class="fas fa-calendar me-1"></i>
                                Published: ${book.published_date}
                            </div>
                        ` : ''}
                        
                        <div class="book-description">
                            ${this.truncateText(book.description || 'No description available.', 150)}
                        </div>
                        
                        ${book.subjects && book.subjects.length > 0 ? `
                            <div class="book-genres">
                                ${book.subjects.slice(0, 3).map(subject => 
                                    `<span class="genre-tag">${subject}</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                        
                        <div class="recommendation-reasons">
                            <h6><i class="fas fa-lightbulb me-1"></i>Why recommended:</h6>
                            <ul>
                                ${reasons.map(reason => `<li>${reason}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Scroll to results
        document.getElementById('recommendations-results').scrollIntoView({
            behavior: 'smooth'
        });
    }

    generateStars(rating) {
        const fullStars = Math.floor(rating);
        const hasHalfStar = rating % 1 >= 0.5;
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
        
        let stars = '';
        
        // Full stars
        for (let i = 0; i < fullStars; i++) {
            stars += '<i class="fas fa-star"></i>';
        }
        
        // Half star
        if (hasHalfStar) {
            stars += '<i class="fas fa-star-half-alt"></i>';
        }
        
        // Empty stars
        for (let i = 0; i < emptyStars; i++) {
            stars += '<i class="far fa-star"></i>';
        }
        
        return stars;
    }

    truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength).trim() + '...';
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification-toast');
        existingNotifications.forEach(notification => notification.remove());
        
        // Create notification
        const notification = document.createElement('div');
        notification.className = `notification-toast alert alert-${this.getBootstrapAlertClass(type)} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            max-width: 500px;
        `;
        
        const icon = this.getNotificationIcon(type);
        
        notification.innerHTML = `
            <i class="${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    getBootstrapAlertClass(type) {
        const mapping = {
            'success': 'success',
            'error': 'danger',
            'warning': 'warning',
            'info': 'info'
        };
        return mapping[type] || 'info';
    }

    getNotificationIcon(type) {
        const mapping = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        return mapping[type] || 'fas fa-info-circle';
    }

    // Book Chat functionality
    async handleBookChatSubmission(e) {
        e.preventDefault();
        
        const input = document.getElementById('book-query');
        const userMessage = input.value.trim();
        
        if (!userMessage) {
            this.showNotification('Please enter a description of what you\'re looking for.', 'error');
            return;
        }

        // Show loading
        document.getElementById('book-loading').style.display = 'block';
        document.getElementById('book-recommendations-results').innerHTML = '';
        
        try {
            const response = await fetch('/api/chat-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: userMessage })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayBookRecommendations(data.recommendations);
                this.showNotification('Book recommendations found!', 'success');
            } else {
                this.displayBookError('Sorry, I couldn\'t find any books matching your description. Try being more specific or using different keywords.');
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error getting book recommendations:', error);
            this.displayBookError('Sorry, I\'m having trouble processing your request right now. Please try again.');
            this.showNotification('Failed to get recommendations. Please try again.', 'error');
        } finally {
            document.getElementById('book-loading').style.display = 'none';
        }
    }

    displayBookRecommendations(recommendations) {
        const container = document.getElementById('book-recommendations-results');
        
        if (!recommendations || recommendations.length === 0) {
            this.displayBookError('No books found matching your description. Try being more specific or using different keywords.');
            return;
        }

        container.innerHTML = `
            <div class="row">
                ${recommendations.map((rec, index) => {
                    const book = rec.book;
                    const reasons = rec.reasons || [];
                    
                    return `
                        <div class="col-lg-4 col-md-6 mb-4">
                            <div class="book-card fade-in-up" style="animation-delay: ${index * 0.2}s">
                                <div class="book-cover">
                                    ${book.cover_url ? 
                                        `<img src="${book.cover_url}" alt="${book.title}" class="book-cover">` :
                                        `<i class="fas fa-book"></i>`
                                    }
                                </div>
                                
                                <div class="book-title">${this.truncateText(book.title, 60)}</div>
                                
                                <div class="book-author">
                                    <i class="fas fa-user me-1"></i>
                                    ${book.authors ? book.authors.join(', ') : 'Unknown Author'}
                                </div>
                                
                                ${book.average_rating ? `
                                    <div class="rating-stars">
                                        ${this.generateStars(book.average_rating)}
                                        <span class="rating-text ms-2">${book.average_rating}/5</span>
                                    </div>
                                ` : ''}
                                
                                ${book.published_date ? `
                                    <div class="text-muted mb-2">
                                        <i class="fas fa-calendar me-1"></i>
                                        Published: ${book.published_date}
                                    </div>
                                ` : ''}
                                
                                <div class="book-description">
                                    ${this.truncateText(book.description || 'No description available.', 150)}
                                </div>
                                
                                ${book.subjects && book.subjects.length > 0 ? `
                                    <div class="book-genres">
                                        ${book.subjects.slice(0, 3).map(subject => 
                                            `<span class="genre-tag">${subject}</span>`
                                        ).join('')}
                                    </div>
                                ` : ''}
                                
                                <div class="recommendation-reasons">
                                    <h6><i class="fas fa-lightbulb me-1"></i>Why this matches:</h6>
                                    <ul>
                                        ${reasons.map(reason => `<li>${reason}</li>`).join('')}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    displayBookError(message) {
        const container = document.getElementById('book-recommendations-results');
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }

    // Chat functionality (legacy)
    async handleChatSubmission(e) {
        e.preventDefault();
        
        const input = document.getElementById('chat-input');
        const userMessage = input.value.trim();
        
        if (!userMessage) {
            this.showNotification('Please enter a description of what you\'re looking for.', 'error');
            return;
        }

        // Add user message to chat
        this.addChatMessage(userMessage, 'user');
        
        // Clear input
        input.value = '';
        
        // Show loading
        this.showChatLoading();
        
        try {
            const response = await fetch('/api/chat-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: userMessage })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayChatRecommendations(data.recommendations);
                this.addChatMessage('I found some great books that match your description!', 'assistant');
                this.showNotification('Recommendations found!', 'success');
            } else {
                this.addChatMessage('Sorry, I couldn\'t find any books matching your description. Try being more specific or using different keywords.', 'assistant');
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error getting chat recommendations:', error);
            this.addChatMessage('Sorry, I\'m having trouble processing your request right now. Please try again.', 'assistant');
            this.showNotification('Failed to get recommendations. Please try again.', 'error');
        } finally {
            this.hideChatLoading();
        }
    }

    addChatMessage(message, sender) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message mb-3`;
        
        const isUser = sender === 'user';
        const avatarClass = isUser ? 'bg-primary' : 'bg-success';
        const avatarIcon = isUser ? 'fas fa-user' : 'fas fa-robot';
        const bubbleClass = isUser ? 'bg-primary text-white ms-auto' : 'bg-light';
        const alignClass = isUser ? 'justify-content-end' : '';
        
        messageDiv.innerHTML = `
            <div class="d-flex align-items-start ${alignClass}">
                ${!isUser ? `
                    <div class="chat-avatar ${avatarClass} text-white rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px; min-width: 40px;">
                        <i class="${avatarIcon}"></i>
                    </div>
                ` : ''}
                <div class="chat-content" style="max-width: 70%;">
                    <div class="chat-bubble ${bubbleClass} p-3 rounded">
                        <p class="mb-0">${message}</p>
                    </div>
                </div>
                ${isUser ? `
                    <div class="chat-avatar ${avatarClass} text-white rounded-circle d-flex align-items-center justify-content-center ms-3" style="width: 40px; height: 40px; min-width: 40px;">
                        <i class="${avatarIcon}"></i>
                    </div>
                ` : ''}
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    showChatLoading() {
        document.getElementById('chat-loading-section').style.display = 'block';
        document.getElementById('chat-recommendations-results').style.display = 'none';
    }

    hideChatLoading() {
        document.getElementById('chat-loading-section').style.display = 'none';
    }

    displayChatRecommendations(recommendations) {
        document.getElementById('chat-loading-section').style.display = 'none';
        document.getElementById('chat-recommendations-results').style.display = 'block';
        
        const container = document.getElementById('chat-recommendations-container');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        No books found matching your description. Try being more specific or using different keywords.
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = recommendations.map((rec, index) => {
            const book = rec.book;
            const reasons = rec.reasons || [];
            
            return `
                <div class="col-lg-4 col-md-6 mb-4">
                    <div class="book-card fade-in-up" style="animation-delay: ${index * 0.2}s">
                        <div class="book-cover">
                            ${book.cover_url ? 
                                `<img src="${book.cover_url}" alt="${book.title}" class="book-cover">` :
                                `<i class="fas fa-book"></i>`
                            }
                        </div>
                        
                        <div class="book-title">${this.truncateText(book.title, 60)}</div>
                        
                        <div class="book-author">
                            <i class="fas fa-user me-1"></i>
                            ${book.authors ? book.authors.join(', ') : 'Unknown Author'}
                        </div>
                        
                        ${book.average_rating ? `
                            <div class="rating-stars">
                                ${this.generateStars(book.average_rating)}
                                <span class="rating-text ms-2">${book.average_rating}/5</span>
                            </div>
                        ` : ''}
                        
                        ${book.published_date ? `
                            <div class="text-muted mb-2">
                                <i class="fas fa-calendar me-1"></i>
                                Published: ${book.published_date}
                            </div>
                        ` : ''}
                        
                        <div class="book-description">
                            ${this.truncateText(book.description || 'No description available.', 150)}
                        </div>
                        
                        ${book.subjects && book.subjects.length > 0 ? `
                            <div class="book-genres">
                                ${book.subjects.slice(0, 3).map(subject => 
                                    `<span class="genre-tag">${subject}</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                        
                        <div class="recommendation-reasons">
                            <h6><i class="fas fa-lightbulb me-1"></i>Why this matches:</h6>
                            <ul>
                                ${reasons.map(reason => `<li>${reason}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Scroll to results
        document.getElementById('chat-recommendations-results').scrollIntoView({
            behavior: 'smooth'
        });
    }

    clearChatRecommendations() {
        document.getElementById('chat-recommendations-results').style.display = 'none';
        document.getElementById('chat-input').focus();
    }

    // Podcast functionality
    startPodcastRecommendations() {
        // Show podcast questionnaire section
        document.getElementById('podcast-questionnaire-section').style.display = 'block';
        document.getElementById('podcast-recommendations-results').style.display = 'none';
        
        // Scroll to recommendations section
        document.getElementById('recommendations').scrollIntoView({
            behavior: 'smooth'
        });
    }

    async handlePodcastFormSubmission(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const preferences = this.extractPodcastPreferences(formData);
        
        // Validate form
        if (!this.validatePodcastPreferences(preferences)) {
            this.showNotification('Please fill in all required fields.', 'error');
            return;
        }

        // Show loading
        this.showPodcastLoading();
        
        try {
            const response = await fetch('/api/podcast-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(preferences)
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayPodcastRecommendations(data.recommendations);
                this.showNotification('Podcast recommendations generated successfully!', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
                this.hidePodcastLoading();
            }
        } catch (error) {
            console.error('Error getting podcast recommendations:', error);
            this.showNotification('Failed to get podcast recommendations. Please try again.', 'error');
            this.hidePodcastLoading();
        }
    }

    extractPodcastPreferences(formData) {
        const preferences = {};
        
        // Extract favorite categories
        const categories = formData.getAll('categories');
        preferences.categories = categories;
        
        // Extract other preferences
        preferences.listening_mood = formData.get('listening_mood');
        preferences.content_type = formData.get('content_type');
        preferences.episode_length = formData.get('episode_length');
        preferences.experience_level = formData.get('experience_level');
        preferences.favorite_hosts = formData.get('favorite_hosts');
        preferences.explicit_content = formData.get('explicit_content');
        
        return preferences;
    }

    validatePodcastPreferences(preferences) {
        // Check required fields
        if (!preferences.listening_mood) {
            return false;
        }
        
        // At least one category should be selected
        if (!preferences.categories || preferences.categories.length === 0) {
            this.showNotification('Please select at least one podcast category.', 'error');
            return false;
        }
        
        return true;
    }

    showPodcastLoading() {
        document.getElementById('podcast-questionnaire-section').style.display = 'none';
        document.getElementById('podcast-loading-section').style.display = 'block';
        document.getElementById('podcast-recommendations-results').style.display = 'none';
    }

    hidePodcastLoading() {
        document.getElementById('podcast-loading-section').style.display = 'none';
        document.getElementById('podcast-questionnaire-section').style.display = 'block';
    }

    displayPodcastRecommendations(recommendations) {
        document.getElementById('podcast-loading-section').style.display = 'none';
        document.getElementById('podcast-questionnaire-section').style.display = 'none';
        document.getElementById('podcast-recommendations-results').style.display = 'block';
        
        const container = document.getElementById('podcast-recommendations-container');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        No podcast recommendations found. Try adjusting your preferences.
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = recommendations.map((podcast, index) => {
            const reason = podcast.recommendation_reason || 'Recommended based on your preferences';
            
            return `
                <div class="col-lg-4 col-md-6 mb-4">
                    <div class="podcast-card fade-in-up" style="animation-delay: ${index * 0.2}s">
                        <div class="podcast-cover">
                            ${podcast.image_url ? 
                                `<img src="${podcast.image_url}" alt="${podcast.title}" class="podcast-cover-img">` :
                                `<i class="fas fa-podcast"></i>`
                            }
                        </div>
                        
                        <div class="podcast-title">${this.truncateText(podcast.title, 60)}</div>
                        
                        <div class="podcast-author">
                            <i class="fas fa-microphone me-1"></i>
                            ${podcast.author || 'Unknown Host'}
                        </div>
                        
                        ${podcast.rating ? `
                            <div class="rating-stars">
                                ${this.generateStars(podcast.rating)}
                                <span class="rating-text ms-2">${podcast.rating}/5</span>
                            </div>
                        ` : ''}
                        
                        ${podcast.episode_count ? `
                            <div class="text-muted mb-2">
                                <i class="fas fa-list me-1"></i>
                                ${podcast.episode_count} episodes
                            </div>
                        ` : ''}
                        
                        ${podcast.explicit ? `
                            <div class="text-warning mb-2">
                                <i class="fas fa-exclamation-triangle me-1"></i>
                                Explicit Content
                            </div>
                        ` : ''}
                        
                        <div class="podcast-description">
                            ${this.truncateText(podcast.description || 'No description available.', 150)}
                        </div>
                        
                        ${podcast.categories && podcast.categories.length > 0 ? `
                            <div class="podcast-categories">
                                ${podcast.categories.slice(0, 3).map(category => 
                                    `<span class="category-tag">${category}</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                        
                        <div class="recommendation-reasons">
                            <h6><i class="fas fa-lightbulb me-1"></i>Why recommended:</h6>
                            <p class="mb-0">${reason}</p>
                        </div>
                        
                        ${podcast.website_url ? `
                            <div class="mt-3">
                                <a href="${podcast.website_url}" target="_blank" class="btn btn-outline-info btn-sm">
                                    <i class="fas fa-external-link-alt me-1"></i>
                                    Listen Now
                                </a>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');

        // Scroll to results
        document.getElementById('podcast-recommendations-results').scrollIntoView({
            behavior: 'smooth'
        });
    }

    // Podcast chat functionality
    async handlePodcastChatSubmission(e) {
        e.preventDefault();
        
        const input = document.getElementById('podcast-query');
        const userMessage = input.value.trim();
        
        if (!userMessage) {
            this.showNotification('Please enter a description of what podcast you\'re looking for.', 'error');
            return;
        }

        // Show loading
        document.getElementById('podcast-loading').style.display = 'block';
        document.getElementById('podcast-recommendations-results').innerHTML = '';
        
        try {
            const response = await fetch('/api/podcast-chat-recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: userMessage })
            });
            
            const data = await response.json();
            
            if (data.status === 'success' && data.recommendations && data.recommendations.length > 0) {
                this.displayPodcastChatRecommendations(data.recommendations);
                this.showNotification('Podcast recommendations found!', 'success');
            } else {
                this.displayPodcastError('Sorry, I couldn\'t find any podcasts matching your description. Try being more specific or using different keywords.');
                if (data.error) {
                    this.showNotification(`Error: ${data.error}`, 'error');
                }
            }
        } catch (error) {
            console.error('Error getting podcast chat recommendations:', error);
            this.displayPodcastError('Sorry, I\'m having trouble processing your request right now. Please try again.');
            this.showNotification('Failed to get podcast recommendations. Please try again.', 'error');
        } finally {
            document.getElementById('podcast-loading').style.display = 'none';
        }
    }

    displayPodcastError(message) {
        const container = document.getElementById('podcast-recommendations-results');
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }

    addPodcastChatMessage(message, sender) {
        const chatContainer = document.getElementById('podcast-chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message mb-3`;
        
        const isUser = sender === 'user';
        const avatarClass = isUser ? 'bg-primary' : 'bg-warning';
        const avatarIcon = isUser ? 'fas fa-user' : 'fas fa-podcast';
        const bubbleClass = isUser ? 'bg-primary text-white ms-auto' : 'bg-light';
        const alignClass = isUser ? 'justify-content-end' : '';
        
        messageDiv.innerHTML = `
            <div class="d-flex align-items-start ${alignClass}">
                ${!isUser ? `
                    <div class="chat-avatar ${avatarClass} text-dark rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px; min-width: 40px;">
                        <i class="${avatarIcon}"></i>
                    </div>
                ` : ''}
                <div class="chat-content" style="max-width: 70%;">
                    <div class="chat-bubble ${bubbleClass} p-3 rounded">
                        <p class="mb-0">${message}</p>
                    </div>
                </div>
                ${isUser ? `
                    <div class="chat-avatar ${avatarClass} text-white rounded-circle d-flex align-items-center justify-content-center ms-3" style="width: 40px; height: 40px; min-width: 40px;">
                        <i class="${avatarIcon}"></i>
                    </div>
                ` : ''}
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    showPodcastChatLoading() {
        document.getElementById('podcast-chat-loading-section').style.display = 'block';
        document.getElementById('podcast-chat-recommendations-results').style.display = 'none';
    }

    hidePodcastChatLoading() {
        document.getElementById('podcast-chat-loading-section').style.display = 'none';
    }

    displayPodcastChatRecommendations(recommendations) {
        document.getElementById('podcast-loading').style.display = 'none';
        
        const container = document.getElementById('podcast-recommendations-results');
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        No podcasts found matching your description. Try being more specific or using different keywords.
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="row">
                ${recommendations.map((podcast, index) => {
                    const reason = podcast.recommendation_reason || 'Recommended based on your description';
                    
                    return `
                        <div class="col-lg-4 col-md-6 mb-4">
                            <div class="podcast-card fade-in-up" style="animation-delay: ${index * 0.2}s">
                                <div class="podcast-cover">
                                    ${podcast.image_url ? 
                                        `<img src="${podcast.image_url}" alt="${podcast.title}" class="podcast-cover-img">` :
                                        `<i class="fas fa-podcast"></i>`
                                    }
                                </div>
                                
                                <div class="podcast-title">${this.truncateText(podcast.title, 60)}</div>
                                
                                <div class="podcast-author">
                                    <i class="fas fa-microphone me-1"></i>
                                    ${podcast.author || 'Unknown Host'}
                                </div>
                                
                                ${podcast.rating ? `
                                    <div class="rating-stars">
                                        ${this.generateStars(podcast.rating)}
                                        <span class="rating-text ms-2">${podcast.rating}/5</span>
                                    </div>
                                ` : ''}
                                
                                ${podcast.episode_count ? `
                                    <div class="text-muted mb-2">
                                        <i class="fas fa-list me-1"></i>
                                        ${podcast.episode_count} episodes
                                    </div>
                                ` : ''}
                                
                                ${podcast.explicit ? `
                                    <div class="text-warning mb-2">
                                        <i class="fas fa-exclamation-triangle me-1"></i>
                                        Explicit Content
                                    </div>
                                ` : ''}
                                
                                <div class="podcast-description">
                                    ${this.truncateText(podcast.description || 'No description available.', 150)}
                                </div>
                                
                                ${podcast.categories && podcast.categories.length > 0 ? `
                                    <div class="podcast-categories">
                                        ${podcast.categories.slice(0, 3).map(category => 
                                            `<span class="category-tag">${category}</span>`
                                        ).join('')}
                                    </div>
                                ` : ''}
                                
                                <div class="recommendation-reasons">
                                    <h6><i class="fas fa-lightbulb me-1"></i>Why this matches:</h6>
                                    <p class="mb-0">${reason}</p>
                                </div>
                                
                                ${podcast.website_url ? `
                                    <div class="mt-3">
                                        <a href="${podcast.website_url}" target="_blank" class="btn btn-outline-warning btn-sm">
                                            <i class="fas fa-external-link-alt me-1"></i>
                                            Listen Now
                                        </a>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;

        // Scroll to results
        container.scrollIntoView({
            behavior: 'smooth'
        });
    }

    clearPodcastChatRecommendations() {
        document.getElementById('podcast-chat-recommendations-results').style.display = 'none';
        document.getElementById('podcast-chat-input').focus();
    }

    // Utility method to format numbers
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
}

// Global functions for HTML onclick handlers
function startRecommendations() {
    window.bookWiseApp.startRecommendations();
}

function loadBooks() {
    window.bookWiseApp.loadBooks();
}

function clearChatRecommendations() {
    window.bookWiseApp.clearChatRecommendations();
}

function startPodcastRecommendations() {
    window.bookWiseApp.startPodcastRecommendations();
}

function clearPodcastChatRecommendations() {
    window.bookWiseApp.clearPodcastChatRecommendations();
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.bookWiseApp = new BookWiseApp();
});

// Add some additional utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add loading states to buttons
function addLoadingState(button, originalText) {
    button.disabled = true;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        Loading...
    `;
    
    return () => {
        button.disabled = false;
        button.innerHTML = originalText;
    };
}

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', function() {
    const animatedElements = document.querySelectorAll('.book-card, .analytics-card');
    animatedElements.forEach(el => observer.observe(el));
});

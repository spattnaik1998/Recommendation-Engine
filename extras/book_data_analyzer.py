#!/usr/bin/env python3
"""
Book Data Visual Analyzer
Provides comprehensive visual analysis of book data imported from Open Library.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BookDataAnalyzer:
    def __init__(self, data_file: str = 'data/books_database.json'):
        """Initialize the analyzer with book data."""
        self.data_file = data_file
        self.books = []
        self.df = None
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self) -> bool:
        """Load book data from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.books = json.load(f)
            
            if not self.books:
                print(f"No books found in {self.data_file}")
                return False
                
            print(f"Loaded {len(self.books)} books from {self.data_file}")
            self._create_dataframe()
            return True
            
        except FileNotFoundError:
            print(f"File {self.data_file} not found!")
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _create_dataframe(self):
        """Create a pandas DataFrame from the book data for easier analysis."""
        data = []
        for book in self.books:
            # Extract publication year
            pub_year = self._extract_year(book.get('first_publish_date', ''))
            
            # Count authors
            author_count = len(book.get('authors', []))
            
            # Count subjects
            subject_count = len(book.get('subjects', []))
            
            # Calculate description length
            desc_length = len(book.get('description', ''))
            
            # Get primary author
            primary_author = book.get('authors', ['Unknown'])[0]
            
            # Get all subjects as a list
            subjects = book.get('subjects', [])
            
            data.append({
                'title': book.get('title', 'Unknown'),
                'primary_author': primary_author,
                'author_count': author_count,
                'publication_year': pub_year,
                'subject_count': subject_count,
                'description_length': desc_length,
                'subjects': subjects,
                'has_cover': len(book.get('covers', [])) > 0
            })
        
        self.df = pd.DataFrame(data)
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from publication date string."""
        if not date_str:
            return 0
        
        # Try to extract 4-digit year
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(date_str))
        if year_match:
            return int(year_match.group(1))
        return 0
    
    def print_basic_statistics(self):
        """Print basic statistics about the book collection."""
        print("\n" + "="*60)
        print("BOOK COLLECTION OVERVIEW")
        print("="*60)
        
        print(f"Total Books: {len(self.books)}")
        print(f"Unique Authors: {self.df['primary_author'].nunique()}")
        print(f"Books with Publication Dates: {sum(1 for year in self.df['publication_year'] if year > 0)}")
        print(f"Books with Descriptions: {sum(1 for length in self.df['description_length'] if length > 0)}")
        print(f"Books with Cover Images: {sum(self.df['has_cover'])}")
        
        # Publication year range
        valid_years = [year for year in self.df['publication_year'] if year > 0]
        if valid_years:
            print(f"Publication Year Range: {min(valid_years)} - {max(valid_years)}")
        
        # Average statistics
        print(f"Average Authors per Book: {self.df['author_count'].mean():.1f}")
        print(f"Average Subjects per Book: {self.df['subject_count'].mean():.1f}")
        print(f"Average Description Length: {self.df['description_length'].mean():.0f} characters")
    
    def analyze_publication_timeline(self):
        """Analyze and visualize publication timeline."""
        print("\n" + "="*60)
        print("PUBLICATION TIMELINE ANALYSIS")
        print("="*60)
        
        # Filter out books without valid publication years
        valid_years = [year for year in self.df['publication_year'] if year > 0]
        
        if not valid_years:
            print("No valid publication years found.")
            return
        
        # Create publication timeline
        plt.figure(figsize=(12, 6))
        
        # Histogram of publication years
        plt.subplot(1, 2, 1)
        plt.hist(valid_years, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Books by Publication Year')
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Books')
        plt.grid(True, alpha=0.3)
        
        # Decade analysis
        decades = [(year // 10) * 10 for year in valid_years]
        decade_counts = Counter(decades)
        
        plt.subplot(1, 2, 2)
        decades_sorted = sorted(decade_counts.keys())
        counts = [decade_counts[decade] for decade in decades_sorted]
        decade_labels = [f"{decade}s" for decade in decades_sorted]
        
        plt.bar(decade_labels, counts, alpha=0.7, color='lightcoral')
        plt.title('Books by Decade')
        plt.xlabel('Decade')
        plt.ylabel('Number of Books')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print decade statistics
        print("\nBooks by Decade:")
        for decade in sorted(decade_counts.keys()):
            print(f"  {decade}s: {decade_counts[decade]} books")
    
    def analyze_authors(self):
        """Analyze author statistics and patterns."""
        print("\n" + "="*60)
        print("AUTHOR ANALYSIS")
        print("="*60)
        
        # Author frequency
        author_counts = self.df['primary_author'].value_counts()
        
        print(f"Total Unique Authors: {len(author_counts)}")
        print(f"Authors with Multiple Books: {sum(1 for count in author_counts if count > 1)}")
        
        # Top authors
        print("\nTop 10 Authors by Number of Books:")
        for i, (author, count) in enumerate(author_counts.head(10).items(), 1):
            print(f"  {i:2d}. {author}: {count} book{'s' if count > 1 else ''}")
        
        # Visualize top authors
        plt.figure(figsize=(12, 8))
        
        # Top 15 authors bar chart
        top_authors = author_counts.head(15)
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(top_authors)), top_authors.values, alpha=0.7, color='lightgreen')
        plt.title('Top 15 Authors by Number of Books')
        plt.xlabel('Authors')
        plt.ylabel('Number of Books')
        plt.xticks(range(len(top_authors)), top_authors.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, top_authors.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(value), ha='center', va='bottom')
        
        # Author productivity distribution
        plt.subplot(2, 1, 2)
        productivity_dist = author_counts.value_counts().sort_index()
        plt.bar(productivity_dist.index, productivity_dist.values, alpha=0.7, color='orange')
        plt.title('Author Productivity Distribution')
        plt.xlabel('Number of Books per Author')
        plt.ylabel('Number of Authors')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_subjects_and_genres(self):
        """Analyze book subjects and genres."""
        print("\n" + "="*60)
        print("SUBJECTS AND GENRES ANALYSIS")
        print("="*60)
        
        # Collect all subjects
        all_subjects = []
        for book in self.books:
            subjects = book.get('subjects', [])
            all_subjects.extend(subjects)
        
        subject_counts = Counter(all_subjects)
        
        print(f"Total Unique Subjects: {len(subject_counts)}")
        print(f"Total Subject Tags: {len(all_subjects)}")
        print(f"Average Subjects per Book: {len(all_subjects) / len(self.books):.1f}")
        
        # Top subjects
        print("\nTop 20 Most Common Subjects:")
        for i, (subject, count) in enumerate(subject_counts.most_common(20), 1):
            print(f"  {i:2d}. {subject}: {count} book{'s' if count > 1 else ''}")
        
        # Visualize subjects
        plt.figure(figsize=(15, 10))
        
        # Top subjects bar chart
        top_subjects = dict(subject_counts.most_common(20))
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(top_subjects)), list(top_subjects.values()), 
                      alpha=0.7, color='mediumpurple')
        plt.title('Top 20 Most Common Subjects')
        plt.xlabel('Subjects')
        plt.ylabel('Number of Books')
        plt.xticks(range(len(top_subjects)), list(top_subjects.keys()), 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Subject frequency distribution
        plt.subplot(2, 2, 2)
        freq_dist = Counter(subject_counts.values())
        plt.bar(freq_dist.keys(), freq_dist.values(), alpha=0.7, color='gold')
        plt.title('Subject Frequency Distribution')
        plt.xlabel('Number of Books per Subject')
        plt.ylabel('Number of Subjects')
        plt.grid(True, alpha=0.3)
        
        # Word cloud style visualization (using bar chart)
        plt.subplot(2, 1, 2)
        top_30_subjects = dict(subject_counts.most_common(30))
        y_pos = np.arange(len(top_30_subjects))
        plt.barh(y_pos, list(top_30_subjects.values()), alpha=0.7, color='lightblue')
        plt.yticks(y_pos, list(top_30_subjects.keys()))
        plt.xlabel('Number of Books')
        plt.title('Top 30 Subjects - Horizontal View')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze genre categories
        self._analyze_genre_categories(subject_counts)
    
    def _analyze_genre_categories(self, subject_counts):
        """Categorize subjects into broader genres."""
        print("\n" + "-"*40)
        print("GENRE CATEGORIZATION")
        print("-"*40)
        
        # Define genre categories
        genre_keywords = {
            'Fiction': ['fiction', 'novel', 'story', 'narrative'],
            'Fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'dragon'],
            'Science Fiction': ['science fiction', 'sci-fi', 'dystopian', 'space', 'technology'],
            'Romance': ['romance', 'love', 'romantic', 'marriage'],
            'Literature': ['literature', 'classic', 'literary'],
            'Adventure': ['adventure', 'quest', 'journey', 'exploration'],
            'Philosophy': ['philosophy', 'philosophical', 'morality', 'ethics'],
            'Historical': ['historical', 'history', 'period'],
            'Children\'s': ['children', 'young adult', 'juvenile'],
            'Crime/Mystery': ['crime', 'mystery', 'detective', 'murder']
        }
        
        # Categorize subjects
        genre_counts = defaultdict(int)
        for subject, count in subject_counts.items():
            subject_lower = subject.lower()
            for genre, keywords in genre_keywords.items():
                if any(keyword in subject_lower for keyword in keywords):
                    genre_counts[genre] += count
                    break
            else:
                genre_counts['Other'] += count
        
        # Display genre statistics
        print("Books by Genre Category:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {count} books")
        
        # Visualize genre distribution
        plt.figure(figsize=(10, 8))
        
        # Pie chart
        plt.subplot(2, 1, 1)
        genres = list(genre_counts.keys())
        counts = list(genre_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
        
        plt.pie(counts, labels=genres, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Genre Distribution')
        
        # Bar chart
        plt.subplot(2, 1, 2)
        bars = plt.bar(genres, counts, alpha=0.7, color=colors)
        plt.title('Books by Genre Category')
        plt.xlabel('Genre')
        plt.ylabel('Number of Books')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_book_characteristics(self):
        """Analyze various book characteristics."""
        print("\n" + "="*60)
        print("BOOK CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        # Description length analysis
        desc_lengths = [len(book.get('description', '')) for book in self.books]
        books_with_desc = [length for length in desc_lengths if length > 0]
        
        print(f"Books with Descriptions: {len(books_with_desc)} ({len(books_with_desc)/len(self.books)*100:.1f}%)")
        if books_with_desc:
            print(f"Average Description Length: {np.mean(books_with_desc):.0f} characters")
            print(f"Median Description Length: {np.median(books_with_desc):.0f} characters")
            print(f"Shortest Description: {min(books_with_desc)} characters")
            print(f"Longest Description: {max(books_with_desc)} characters")
        
        # Subject count analysis
        subject_counts = [len(book.get('subjects', [])) for book in self.books]
        print(f"\nSubject Count Statistics:")
        print(f"Average Subjects per Book: {np.mean(subject_counts):.1f}")
        print(f"Books with No Subjects: {subject_counts.count(0)}")
        print(f"Maximum Subjects on a Book: {max(subject_counts)}")
        
        # Visualize characteristics
        plt.figure(figsize=(15, 10))
        
        # Description length distribution
        plt.subplot(2, 3, 1)
        if books_with_desc:
            plt.hist(books_with_desc, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Description Length Distribution')
        plt.xlabel('Description Length (characters)')
        plt.ylabel('Number of Books')
        plt.grid(True, alpha=0.3)
        
        # Subject count distribution
        plt.subplot(2, 3, 2)
        plt.hist(subject_counts, bins=range(max(subject_counts)+2), alpha=0.7, 
                color='lightgreen', edgecolor='black')
        plt.title('Subject Count Distribution')
        plt.xlabel('Number of Subjects')
        plt.ylabel('Number of Books')
        plt.grid(True, alpha=0.3)
        
        # Cover availability
        plt.subplot(2, 3, 3)
        cover_counts = [1 if book.get('covers') else 0 for book in self.books]
        labels = ['With Cover', 'Without Cover']
        sizes = [sum(cover_counts), len(cover_counts) - sum(cover_counts)]
        colors = ['lightblue', 'lightgray']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Cover Image Availability')
        
        # Publication year vs description length (if both available)
        valid_data = [(year, length) for year, length in 
                     zip(self.df['publication_year'], self.df['description_length'])
                     if year > 0 and length > 0]
        
        if valid_data:
            years, lengths = zip(*valid_data)
            plt.subplot(2, 3, 4)
            plt.scatter(years, lengths, alpha=0.6, color='purple')
            plt.title('Publication Year vs Description Length')
            plt.xlabel('Publication Year')
            plt.ylabel('Description Length')
            plt.grid(True, alpha=0.3)
        
        # Author count distribution
        plt.subplot(2, 3, 5)
        author_counts = self.df['author_count'].value_counts().sort_index()
        plt.bar(author_counts.index, author_counts.values, alpha=0.7, color='orange')
        plt.title('Number of Authors per Book')
        plt.xlabel('Number of Authors')
        plt.ylabel('Number of Books')
        plt.grid(True, alpha=0.3)
        
        # Title length analysis
        title_lengths = [len(book.get('title', '')) for book in self.books]
        plt.subplot(2, 3, 6)
        plt.hist(title_lengths, bins=15, alpha=0.7, color='gold', edgecolor='black')
        plt.title('Title Length Distribution')
        plt.xlabel('Title Length (characters)')
        plt.ylabel('Number of Books')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_reading_recommendations(self):
        """Generate reading recommendations based on data analysis."""
        print("\n" + "="*60)
        print("READING RECOMMENDATIONS")
        print("="*60)
        
        # Most popular subjects
        all_subjects = []
        for book in self.books:
            all_subjects.extend(book.get('subjects', []))
        
        popular_subjects = Counter(all_subjects).most_common(5)
        
        print("📚 Based on your collection, you might enjoy books in these popular genres:")
        for i, (subject, count) in enumerate(popular_subjects, 1):
            print(f"  {i}. {subject} ({count} books in collection)")
        
        # Prolific authors
        author_counts = self.df['primary_author'].value_counts()
        prolific_authors = author_counts[author_counts > 1].head(5)
        
        if not prolific_authors.empty:
            print(f"\n📖 Authors with multiple books in your collection:")
            for author, count in prolific_authors.items():
                print(f"  • {author} ({count} books)")
        
        # Time period recommendations
        valid_years = [year for year in self.df['publication_year'] if year > 0]
        if valid_years:
            decade_counts = Counter([(year // 10) * 10 for year in valid_years])
            popular_decade = max(decade_counts.items(), key=lambda x: x[1])
            print(f"\n📅 Your collection is strong in {popular_decade[0]}s literature ({popular_decade[1]} books)")
        
        # Books with rich descriptions
        books_with_long_desc = [(book['title'], len(book.get('description', ''))) 
                               for book in self.books 
                               if len(book.get('description', '')) > 200]
        
        if books_with_long_desc:
            books_with_long_desc.sort(key=lambda x: x[1], reverse=True)
            print(f"\n📝 Books with detailed descriptions (great for browsing):")
            for title, length in books_with_long_desc[:5]:
                print(f"  • {title} ({length} characters)")
    
    def run_complete_analysis(self):
        """Run the complete analysis suite."""
        print("🔍 Starting comprehensive book data analysis...")
        
        if not self.load_data():
            return
        
        # Run all analyses
        self.print_basic_statistics()
        self.analyze_publication_timeline()
        self.analyze_authors()
        self.analyze_subjects_and_genres()
        self.analyze_book_characteristics()
        self.generate_reading_recommendations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("All visualizations have been displayed.")
        print("Check the charts and statistics above for insights into your book collection.")


def main():
    """Main function to run the book data analyzer."""
    analyzer = BookDataAnalyzer()
    
    # Check for available data files, prioritizing API data
    import os
    if os.path.exists('data/books_database.json'):
        analyzer.data_file = 'data/books_database.json'
        print("Using API scraped book data: data/books_database.json")
    elif os.path.exists('imported_books.json'):
        analyzer.data_file = 'imported_books.json'
        print("Using imported_books.json data file")
    else:
        print("❌ No book data found! Please run the data loader first.")
        print("Run: python -c 'from backend.data_loader import DataLoader; DataLoader().load_books_from_api()'")
        return
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()

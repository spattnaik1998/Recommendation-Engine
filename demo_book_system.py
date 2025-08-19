#!/usr/bin/env python3
"""
Demo Script for Book Analysis and Recommendation System
Demonstrates the complete book data analysis and RAG-based recommendation pipeline.
"""

import os
import json
from book_data_analyzer import BookDataAnalyzer
from book_recommender_rag import BookRecommenderRAG

def main():
    """Main demo function."""
    print("🎉 Welcome to the Complete Book Analysis & Recommendation System!")
    print("="*70)
    
    # Check available data files
    data_file = 'data/books_database.json'
    if os.path.exists('data/books_database.json'):
        print(f"📚 Using API scraped book data: {data_file}")
    elif os.path.exists('imported_books.json'):
        data_file = 'imported_books.json'
        print(f"📚 Using imported book data: {data_file}")
    else:
        print("❌ No book data found! Please run the data loader first.")
        print("Run: python -c 'from backend.data_loader import DataLoader; DataLoader().load_books_from_api()'")
        return
    
    print("\nThis demo will show you:")
    print("1. 📊 Comprehensive book data analysis with visualizations")
    print("2. 🤖 AI-powered book recommendations based on your preferences")
    print("3. 💾 Persistent user profiles for future recommendations")
    
    # Ask user what they want to do
    print("\n" + "="*70)
    print("What would you like to do?")
    print("1. Run Book Data Analysis (charts and statistics)")
    print("2. Get Personalized Book Recommendations")
    print("3. Run Both (complete demo)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_analysis_demo(data_file)
        elif choice == '2':
            run_recommendation_demo(data_file)
        elif choice == '3':
            run_complete_demo(data_file)
        elif choice == '4':
            print("👋 Goodbye! Thanks for trying our book system!")
        else:
            print("❌ Invalid choice. Please run the demo again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

def run_analysis_demo(data_file):
    """Run the book data analysis demo."""
    print("\n🔍 Starting Book Data Analysis...")
    print("="*50)
    
    analyzer = BookDataAnalyzer(data_file)
    analyzer.run_complete_analysis()
    
    print("\n✅ Analysis complete! Check the visualizations above.")

def run_recommendation_demo(data_file):
    """Run the book recommendation demo."""
    print("\n🤖 Starting Book Recommendation System...")
    print("="*50)
    
    recommender = BookRecommenderRAG(data_file)
    recommender.run_recommendation_system()
    
    print("\n✅ Recommendations complete!")

def run_complete_demo(data_file):
    """Run both analysis and recommendation demos."""
    print("\n🚀 Running Complete Book System Demo...")
    print("="*50)
    
    # First run analysis
    print("\n📊 PART 1: BOOK DATA ANALYSIS")
    print("-" * 40)
    run_analysis_demo(data_file)
    
    # Ask if user wants to continue to recommendations
    print("\n" + "="*50)
    continue_choice = input("Continue to book recommendations? (y/n): ").strip().lower()
    
    if continue_choice == 'y':
        print("\n🎯 PART 2: PERSONALIZED RECOMMENDATIONS")
        print("-" * 40)
        run_recommendation_demo(data_file)
    else:
        print("✅ Demo complete! You can run recommendations anytime.")

def show_system_info():
    """Show information about the system capabilities."""
    print("\n📋 SYSTEM CAPABILITIES:")
    print("="*50)
    
    print("\n📊 Book Data Analysis Features:")
    print("  • Collection overview and statistics")
    print("  • Publication timeline analysis")
    print("  • Author productivity analysis")
    print("  • Genre and subject analysis")
    print("  • Book characteristics analysis")
    print("  • Visual charts and graphs")
    print("  • Reading recommendations based on collection")
    
    print("\n🤖 RAG-Based Recommendation Features:")
    print("  • Interactive preference questionnaire")
    print("  • Content-based filtering using TF-IDF")
    print("  • Hybrid recommendation approach")
    print("  • Personalized explanation for each recommendation")
    print("  • Persistent user profiles")
    print("  • Genre, mood, and time period preferences")
    print("  • Author diversity in recommendations")
    
    print("\n🔧 Technical Features:")
    print("  • JSON-based book data storage")
    print("  • Scikit-learn for machine learning")
    print("  • Matplotlib/Seaborn for visualizations")
    print("  • Pandas for data manipulation")
    print("  • TF-IDF vectorization for content similarity")
    print("  • Cosine similarity for recommendations")

if __name__ == "__main__":
    # Show system info first
    show_system_info()
    
    # Run main demo
    main()

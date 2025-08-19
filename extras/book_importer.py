#!/usr/bin/env python3
"""
Open Library Book Importer
Downloads and processes Open Library data dumps to import 1000 books with author information.
"""

import gzip
import json
import requests
import os
from typing import Dict, List, Optional
import time

class OpenLibraryImporter:
    def __init__(self):
        self.works_url = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
        self.authors_url = "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz"
        self.works_file = "ol_dump_works_latest.txt.gz"
        self.authors_file = "ol_dump_authors_latest.txt.gz"
        self.books = []
        self.authors_cache = {}
        
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL if it doesn't exist locally."""
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download.")
            return True
            
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            print(f"\n{filename} downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    def load_authors_cache(self, limit: int = 10000) -> None:
        """Load author information into cache for faster lookup."""
        print("Loading authors cache...")
        count = 0
        
        try:
            with gzip.open(self.authors_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if count >= limit:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        if data.get('type', {}).get('key') == '/type/author':
                            author_key = data.get('key', '')
                            author_name = data.get('name', 'Unknown Author')
                            self.authors_cache[author_key] = author_name
                            count += 1
                            
                            if count % 1000 == 0:
                                print(f"Loaded {count} authors...")
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error loading authors: {e}")
            
        print(f"Authors cache loaded: {len(self.authors_cache)} authors")
    
    def get_author_name(self, author_key: str) -> str:
        """Get author name from cache or return default."""
        return self.authors_cache.get(author_key, "Unknown Author")
    
    def extract_books(self, target_count: int = 1000) -> List[Dict]:
        """Extract books from the works data dump."""
        print(f"Extracting {target_count} books...")
        books = []
        processed = 0
        
        try:
            with gzip.open(self.works_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if len(books) >= target_count:
                        break
                        
                    processed += 1
                    if processed % 10000 == 0:
                        print(f"Processed {processed} records, found {len(books)} books...")
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Check if this is a work (book)
                        if data.get('type', {}).get('key') == '/type/work':
                            book_info = self.extract_book_info(data)
                            if book_info:
                                books.append(book_info)
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error extracting books: {e}")
            
        print(f"Extraction complete: {len(books)} books found")
        return books
    
    def extract_book_info(self, work_data: Dict) -> Optional[Dict]:
        """Extract relevant information from a work record."""
        try:
            # Basic book information
            title = work_data.get('title', 'Unknown Title')
            key = work_data.get('key', '')
            
            # Skip if no title
            if not title or title == 'Unknown Title':
                return None
            
            # Extract authors
            authors = []
            author_data = work_data.get('authors', [])
            for author in author_data:
                if isinstance(author, dict):
                    author_key = author.get('author', {}).get('key', '') if 'author' in author else author.get('key', '')
                    if author_key:
                        author_name = self.get_author_name(author_key)
                        authors.append(author_name)
            
            # Extract other information
            description = work_data.get('description', '')
            if isinstance(description, dict):
                description = description.get('value', '')
            elif isinstance(description, list) and description:
                description = description[0] if isinstance(description[0], str) else description[0].get('value', '')
            
            subjects = work_data.get('subjects', [])
            if isinstance(subjects, list):
                subjects = [s for s in subjects if isinstance(s, str)][:5]  # Limit to 5 subjects
            
            first_publish_date = work_data.get('first_publish_date', '')
            
            book_info = {
                'key': key,
                'title': title,
                'authors': authors if authors else ['Unknown Author'],
                'description': description[:500] if description else '',  # Limit description length
                'subjects': subjects,
                'first_publish_date': first_publish_date,
                'covers': work_data.get('covers', [])
            }
            
            return book_info
            
        except Exception as e:
            print(f"Error extracting book info: {e}")
            return None
    
    def save_books_to_json(self, books: List[Dict], filename: str = 'imported_books.json') -> None:
        """Save books to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(books, f, indent=2, ensure_ascii=False)
            print(f"Books saved to {filename}")
        except Exception as e:
            print(f"Error saving books: {e}")
    
    def print_imported_books(self, books: List[Dict], count: int = 5) -> None:
        """Print imported books for verification."""
        print(f"\n--- Sample of {min(count, len(books))} books ---")
        for i, book in enumerate(books[:count]):
            print(f"\n{i+1}. Title: {book['title']}")
            print(f"   Authors: {', '.join(book['authors'])}")
            print(f"   Key: {book['key']}")
            if book['first_publish_date']:
                print(f"   First Published: {book['first_publish_date']}")
            if book['subjects']:
                print(f"   Subjects: {', '.join(book['subjects'][:3])}")
            if book['description']:
                desc = book['description'][:100] + "..." if len(book['description']) > 100 else book['description']
                print(f"   Description: {desc}")
    
    def run(self, target_books: int = 1000) -> None:
        """Main execution method."""
        print("=== Open Library Book Importer ===")
        print(f"Target: Import {target_books} books\n")
        
        # Step 1: Download data files
        print("Step 1: Downloading data files...")
        if not self.download_file(self.works_url, self.works_file):
            print("Failed to download works file. Exiting.")
            return
            
        if not self.download_file(self.authors_url, self.authors_file):
            print("Failed to download authors file. Exiting.")
            return
        
        # Step 2: Load authors cache
        print("\nStep 2: Loading authors cache...")
        self.load_authors_cache()
        
        # Step 3: Extract books
        print(f"\nStep 3: Extracting {target_books} books...")
        books = self.extract_books(target_books)
        
        if not books:
            print("No books were extracted. Exiting.")
            return
        
        # Step 4: Save results
        print(f"\nStep 4: Saving {len(books)} books...")
        self.save_books_to_json(books)
        
        # Step 5: Display sample
        print(f"\nStep 5: Results summary")
        print(f"Successfully imported {len(books)} books!")
        self.print_imported_books(books)
        
        print(f"\n=== Import Complete ===")
        print(f"Total books imported: {len(books)}")
        print(f"Data saved to: imported_books.json")


def main():
    """Main function to run the importer."""
    importer = OpenLibraryImporter()
    importer.run(1000)


if __name__ == "__main__":
    main()

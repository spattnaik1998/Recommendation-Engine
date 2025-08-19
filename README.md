# Open Library Book Importer

This Python script downloads and processes Open Library data dumps to import 1000 books with author information.

## Features

- Downloads Open Library works and authors data dumps
- Extracts 1000 books with complete metadata
- Matches books with author information
- Handles compressed (.gz) files efficiently
- Saves results to JSON format
- Provides progress tracking and error handling

## Requirements

- Python 3.6+
- requests library

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python book_importer.py
```

The script will:
1. Download the latest Open Library data dumps (~2GB total)
2. Load author information into cache
3. Extract 1000 books from the works data
4. Match books with author names
5. Save results to `imported_books.json`

## Output

The script generates `imported_books.json` containing an array of book objects with:
- `key`: Open Library identifier
- `title`: Book title
- `authors`: List of author names
- `description`: Book description (truncated to 500 chars)
- `subjects`: List of subject tags
- `first_publish_date`: Publication date
- `covers`: Cover image IDs

## Data Sources

- Works: https://openlibrary.org/data/ol_dump_works_latest.txt.gz
- Authors: https://openlibrary.org/data/ol_dump_authors_latest.txt.gz

## Notes

- The data dumps are large files (~1GB each compressed)
- First run will take time to download the files
- Subsequent runs will use cached files
- The script processes data efficiently using streaming

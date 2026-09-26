"""
Extract text content from PDF files.
Handles academic papers and attempts to extract metadata.
"""

import re
from datetime import datetime
from pathlib import Path
import PyPDF2
import pdfplumber


class PDFTextExtractor:
    """Extract clean text and metadata from PDF files."""
    
    def __init__(self):
        self.date_patterns = [
            r'\b(19|20)\d{2}\b',  # Year only
            r'\d{1,2}[\s\-/]\d{1,2}[\s\-/](19|20)\d{2}',  # Various date formats
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(19|20)\d{2}',
        ]
    
    def extract_from_file(self, pdf_path, use_pdfplumber=True):
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_pdfplumber: If True, try pdfplumber first (better text extraction)
            
        Returns:
            dict with 'text', 'title', 'date', 'author', 'metadata'
        """
        try:
            # Try pdfplumber first (better for complex layouts)
            if use_pdfplumber:
                result = self._extract_with_pdfplumber(pdf_path)
                if result['success']:
                    return result
            
            # Fallback to PyPDF2
            return self._extract_with_pypdf2(pdf_path)
            
        except Exception as e:
            return {
                'text': '',
                'title': '',
                'date': None,
                'author': None,
                'metadata': {},
                'success': False,
                'error': str(e)
            }
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (better for complex PDFs)."""
        try:
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                # Get metadata
                metadata = pdf.metadata or {}
                
                # Extract text from all pages
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = '\n\n'.join(text_parts)
            
            # Extract metadata
            title = self._extract_title_from_metadata(metadata) or self._extract_title_from_text(full_text)
            date = self._extract_date_from_metadata(metadata) or self._extract_date_from_text(full_text)
            author = self._extract_author_from_metadata(metadata) or self._extract_author_from_text(full_text)
            
            # Clean text
            clean_text = self._clean_text(full_text)
            
            # Create formatted text with date as first line
            formatted_text = self._format_text_with_metadata(clean_text, date, title, author)
            
            return {
                'text': clean_text,
                'title': title,
                'date': date,
                'author': author,
                'metadata': metadata,
                'formatted_text': formatted_text,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'title': '',
                'date': None,
                'author': None,
                'metadata': {},
                'success': False,
                'error': f"pdfplumber error: {str(e)}"
            }
    
    def _extract_with_pypdf2(self, pdf_path):
        """Extract text using PyPDF2 (fallback method)."""
        try:
            text_parts = []
            metadata = {}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get metadata
                if pdf_reader.metadata:
                    metadata = {
                        key.replace('/', ''): value 
                        for key, value in pdf_reader.metadata.items()
                    }
                
                # Extract text from all pages
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = '\n\n'.join(text_parts)
            
            # Extract metadata
            title = self._extract_title_from_metadata(metadata) or self._extract_title_from_text(full_text)
            date = self._extract_date_from_metadata(metadata) or self._extract_date_from_text(full_text)
            author = self._extract_author_from_metadata(metadata) or self._extract_author_from_text(full_text)
            
            # Clean text
            clean_text = self._clean_text(full_text)
            
            # Create formatted text with date as first line
            formatted_text = self._format_text_with_metadata(clean_text, date, title, author)
            
            return {
                'text': clean_text,
                'title': title,
                'date': date,
                'author': author,
                'metadata': metadata,
                'formatted_text': formatted_text,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'title': '',
                'date': None,
                'author': None,
                'metadata': {},
                'success': False,
                'error': f"PyPDF2 error: {str(e)}"
            }
    
    def _extract_title_from_metadata(self, metadata):
        """Extract title from PDF metadata."""
        if not metadata:
            return None
        
        # Try common title fields
        title_fields = ['Title', 'title', 'Subject', 'subject']
        for field in title_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field]).strip()
        
        return None
    
    def _extract_title_from_text(self, text):
        """Extract title from PDF text (usually first line or bold text)."""
        if not text:
            return None
        
        # Get first few lines
        lines = text.strip().split('\n')
        
        # First non-empty line is often the title
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip common headers
                if not any(skip in line.lower() for skip in ['page', 'doi:', 'arxiv:', 'abstract']):
                    return line
        
        return None
    
    def _extract_date_from_metadata(self, metadata):
        """Extract date from PDF metadata."""
        if not metadata:
            return None
        
        # Try common date fields
        date_fields = ['CreationDate', 'ModDate', 'Date', 'date']
        for field in date_fields:
            if field in metadata and metadata[field]:
                date_str = str(metadata[field])
                parsed_date = self._parse_pdf_date(date_str)
                if parsed_date:
                    return parsed_date
        
        return None
    
    def _extract_date_from_text(self, text):
        """Extract date from PDF text."""
        if not text:
            return None
        
        # Search first few pages for dates
        first_page = text[:2000]  # First ~2000 chars
        
        # Look for date patterns
        for pattern in self.date_patterns:
            matches = re.findall(pattern, first_page, re.I)
            if matches:
                # Try to parse the date
                date_match = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                parsed_date = self._parse_date_string(date_match)
                if parsed_date:
                    return parsed_date
        
        return None
    
    def _parse_pdf_date(self, date_str):
        """Parse PDF metadata date format (D:YYYYMMDDHHmmSS)."""
        if not date_str:
            return None
        
        # PDF date format: D:YYYYMMDDHHmmSS+HH'mm'
        pdf_date_match = re.search(r'D:(\d{4})(\d{2})(\d{2})', date_str)
        if pdf_date_match:
            year, month, day = pdf_date_match.groups()
            try:
                return f"{year}-{month}-{day}"
            except:
                pass
        
        # Try standard date parsing
        return self._parse_date_string(date_str)
    
    def _parse_date_string(self, date_str):
        """Parse various date formats into YYYY-MM-DD."""
        if not date_str:
            return None
        
        # Just a year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = year_match.group(0)
            # Return year with January 1st as default
            return f"{year}-01-01"
        
        return None
    
    def _extract_author_from_metadata(self, metadata):
        """Extract author from PDF metadata."""
        if not metadata:
            return None
        
        # Try common author fields
        author_fields = ['Author', 'author', 'Creator', 'creator']
        for field in author_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field]).strip()
        
        return None
    
    def _extract_author_from_text(self, text):
        """Extract author from PDF text."""
        if not text:
            return None
        
        # Get first page text
        first_page = text[:2000]
        
        # Look for author patterns (after title, before abstract)
        lines = first_page.split('\n')
        
        # Look for lines that might be authors (between title and abstract)
        in_author_section = False
        potential_authors = []
        
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Stop at abstract
            if 'abstract' in line.lower():
                break
            
            # After first line (title), look for names
            if i > 0:
                # Check if line looks like author names
                if self._looks_like_author(line):
                    potential_authors.append(line)
                    if len(potential_authors) >= 3:  # Max 3 author lines
                        break
        
        if potential_authors:
            return ', '.join(potential_authors)
        
        return None
    
    def _looks_like_author(self, line):
        """Check if a line looks like author names."""
        # Should have reasonable length
        if len(line) < 5 or len(line) > 100:
            return False
        
        # Should not be all caps (likely header)
        if line.isupper():
            return False
        
        # Should have typical name patterns
        name_indicators = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # FirstName LastName
            r'[A-Z]\.\s*[A-Z]\.',  # Initials
            r'\band\b',  # "and" between names
        ]
        
        for indicator in name_indicators:
            if re.search(indicator, line):
                return True
        
        return False
    
    def _clean_text(self, text):
        """Clean extracted text."""
        if not text:
            return ''
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common pattern: lone numbers)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove headers/footers (repeated text)
        # This is complex, so we'll skip for now
        
        return text.strip()
    
    def _format_text_with_metadata(self, text, date, title, author):
        """
        Format text with metadata header.
        Date will be the first line as requested.
        """
        lines = []
        
        # Date as first line (required)
        if date:
            lines.append(f"Publication Date: {date}")
        else:
            lines.append("Publication Date: Unknown")
        
        # Add other metadata
        if title:
            lines.append(f"Title: {title}")
        if author:
            lines.append(f"Author: {author}")
        
        # Separator
        lines.append("\n" + "="*80 + "\n")
        
        # Main text
        lines.append(text)
        
        return "\n".join(lines)


def extract_pdf_text(pdf_path, use_pdfplumber=True):
    """
    Convenience function to extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file (string or Path object)
        use_pdfplumber: If True, try pdfplumber first
        
    Returns:
        dict with extraction results
    """
    extractor = PDFTextExtractor()
    return extractor.extract_from_file(pdf_path, use_pdfplumber)


if __name__ == '__main__':
    # Test the extractor
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        result = extract_pdf_text(test_file)
        
        print("=" * 80)
        print(f"Title: {result['title']}")
        print(f"Date: {result['date']}")
        print(f"Author: {result['author']}")
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("=" * 80)
        print("\nMetadata:")
        for key, value in result.get('metadata', {}).items():
            print(f"  {key}: {value}")
        print("\n" + "=" * 80)
        print("\nText Preview (first 500 chars):")
        print(result['text'][:500])
        print("\n" + "=" * 80)
    else:
        print("Usage: python extract_pdf_text.py <pdf_file>")

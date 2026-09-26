"""
Extract text content from HTML files.
Handles saved web pages and attempts to extract publication dates.
"""

from bs4 import BeautifulSoup
import re
from datetime import datetime
from pathlib import Path
import json


class HTMLTextExtractor:
    """Extract clean text and metadata from HTML files."""
    
    def __init__(self):
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2023-08-08
            r'\d{1,2}/\d{1,2}/\d{4}',  # 08/08/2023
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # 8 Aug 2023
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # Aug 8, 2023
        ]
    
    def extract_from_file(self, html_path):
        """
        Extract text and metadata from an HTML file.
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            dict with 'text', 'title', 'date', 'author', 'url', 'formatted_text'
        """
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            title = self._extract_title(soup)
            date = self._extract_date(soup, html_content)
            author = self._extract_author(soup)
            url = self._extract_url(soup)
            
            # Extract main content
            text = self._extract_text(soup)
            
            # Create formatted text with date as first line
            formatted_text = self._format_text_with_metadata(text, date, title, author, url)
            
            return {
                'text': text,
                'title': title,
                'date': date,
                'author': author,
                'url': url,
                'formatted_text': formatted_text,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'title': '',
                'date': None,
                'author': None,
                'url': None,
                'success': False,
                'error': str(e)
            }
    
    def _extract_title(self, soup):
        """Extract title from HTML."""
        # Try meta title first
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        
        # Try title tag
        if soup.title:
            title = soup.title.string
            if title:
                # Clean up title (remove site name if present)
                title = title.strip()
                # Remove common patterns like " | InfoWorld" or " - ACM Queue"
                title = re.split(r'\s+[\|\-]\s+', title)[0]
                return title
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        return ''
    
    def _extract_date(self, soup, html_content):
        """Extract publication date from HTML."""
        # Try meta tags first
        date_meta_tags = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'datePublished'}),
            ('meta', {'name': 'date'}),
            ('meta', {'property': 'og:published_time'}),
            ('meta', {'name': 'publication_date'}),
            ('meta', {'itemprop': 'datePublished'}),
            ('time', {'datetime': True}),
            ('time', {'pubdate': True}),
        ]
        
        for tag_name, attrs in date_meta_tags:
            tag = soup.find(tag_name, attrs)
            if tag:
                # Get content from various possible attributes
                date_str = (tag.get('content') or 
                           tag.get('datetime') or 
                           tag.get_text())
                if date_str:
                    parsed_date = self._parse_date_string(date_str.strip())
                    if parsed_date:
                        return parsed_date
        
        # Look for date in common CSS classes
        date_classes = ['date', 'published', 'post-date', 'article-date', 
                       'byline', 'timestamp', 'publish-date']
        for class_name in date_classes:
            elements = soup.find_all(class_=re.compile(class_name, re.I))
            for elem in elements:
                text = elem.get_text().strip()
                parsed_date = self._parse_date_string(text)
                if parsed_date:
                    return parsed_date
        
        # Search entire content for date patterns
        for pattern in self.date_patterns:
            matches = re.findall(pattern, html_content, re.I)
            if matches:
                # Take the first match
                parsed_date = self._parse_date_string(matches[0])
                if parsed_date:
                    return parsed_date
        
        return None
    
    def _parse_date_string(self, date_str):
        """Parse various date formats into YYYY-MM-DD."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to extract just the date part if it's ISO format
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
        if iso_match:
            return iso_match.group(1)
        
        return None
    
    def _extract_author(self, soup):
        """Extract author from HTML."""
        # Try meta tags
        meta_author_tags = [
            ('meta', {'name': 'author'}),
            ('meta', {'property': 'article:author'}),
            ('meta', {'name': 'twitter:creator'}),
        ]
        
        for tag_name, attrs in meta_author_tags:
            tag = soup.find(tag_name, attrs)
            if tag and tag.get('content'):
                author = tag['content'].strip()
                if author and author.lower() != 'no author':
                    return author
        
        # Look for author in common CSS classes
        author_classes = ['author', 'by', 'byline', 'writer']
        for class_name in author_classes:
            elements = soup.find_all(class_=re.compile(class_name, re.I))
            for elem in elements:
                text = elem.get_text().strip()
                # Clean up common prefixes
                text = re.sub(r'^(by|author|written by)[\s:]+', '', text, flags=re.I)
                if text and len(text) < 100:  # Reasonable author name length
                    return text
        
        return None
    
    def _extract_url(self, soup):
        """Extract canonical URL from HTML."""
        # Try canonical link
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            return canonical['href']
        
        # Try meta og:url
        og_url = soup.find('meta', property='og:url')
        if og_url and og_url.get('content'):
            return og_url['content']
        
        return None
    
    def _extract_text(self, soup):
        """Extract main text content from HTML."""
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'iframe', 'noscript']):
            script.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Look for common article containers
        content_selectors = [
            ('article', {}),
            ('div', {'class': re.compile(r'article|content|post|entry|main', re.I)}),
            ('main', {}),
        ]
        
        for tag_name, attrs in content_selectors:
            main_content = soup.find(tag_name, attrs)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            main_content = soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text):
        """Clean extracted text."""
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove common web artifacts
        artifacts = [
            r'Advertisement',
            r'Cookie Policy',
            r'Privacy Policy',
            r'Terms of Service',
            r'Subscribe to Newsletter',
            r'Sign up for',
        ]
        
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.I)
        
        return text.strip()
    
    def _format_text_with_metadata(self, text, date, title, author, url):
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
        if url:
            lines.append(f"URL: {url}")
        
        # Separator
        lines.append("\n" + "="*80 + "\n")
        
        # Main text
        lines.append(text)
        
        return "\n".join(lines)


def extract_html_text(html_path):
    """
    Convenience function to extract text from an HTML file.
    
    Args:
        html_path: Path to HTML file (string or Path object)
        
    Returns:
        dict with extraction results
    """
    extractor = HTMLTextExtractor()
    return extractor.extract_from_file(html_path)


if __name__ == '__main__':
    # Test the extractor
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        result = extract_html_text(test_file)
        
        print("=" * 80)
        print(f"Title: {result['title']}")
        print(f"Date: {result['date']}")
        print(f"Author: {result['author']}")
        print(f"URL: {result['url']}")
        print("=" * 80)
        print("\nFormatted Text Preview (first 800 chars - showing date as first line):")
        print(result.get('formatted_text', result['text'])[:800])
        print("\n" + "=" * 80)
    else:
        print("Usage: python extract_html_text.py <html_file>")

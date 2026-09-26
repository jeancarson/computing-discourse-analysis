"""
Process ERA_2 sources and create Era2 directory structure with cleaned text files.
This script processes all HTML and PDF files from ERA_2 sources, extracts text,
and creates a structured directory similar to Era1.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Import our extraction modules
sys.path.append(os.path.dirname(__file__))
from extract_html_text import extract_html_text
from extract_pdf_text import extract_pdf_text


class Era2Processor:
    """Process ERA_2 sources and create Era2 directory structure."""
    
    def __init__(self, era2_source_path, era2_output_path):
        """
        Initialize the processor.
        
        Args:
            era2_source_path: Path to Sources/Papers/ERA_2
            era2_output_path: Path to Sources/CleanedSources/Era2 (will be created)
        """
        self.era2_source_path = Path(era2_source_path)
        self.era2_output_path = Path(era2_output_path)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'by_source': {}
        }
        
        # Error log
        self.errors = []
    
    def process_all_sources(self):
        """Process all source directories in ERA_2."""
        print("="*80)
        print("ERA_2 Processing - Stage 2")
        print("="*80)
        print(f"Source: {self.era2_source_path}")
        print(f"Output: {self.era2_output_path}")
        print("="*80)
        
        # Create base Era2 directory
        self.era2_output_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created Era2 directory: {self.era2_output_path}")
        
        # Get all source subdirectories
        source_dirs = [d for d in self.era2_source_path.iterdir() if d.is_dir()]
        
        print(f"\nFound {len(source_dirs)} source directories to process:")
        for source_dir in source_dirs:
            print(f"  - {source_dir.name}")
        
        print("\n" + "="*80)
        
        # Process each source directory
        for source_dir in source_dirs:
            self.process_source_directory(source_dir)
        
        # Print final statistics
        self.print_statistics()
        
        # Save error log if any errors occurred
        if self.errors:
            self.save_error_log()
    
    def process_source_directory(self, source_dir):
        """Process a single source directory."""
        source_name = source_dir.name
        print(f"\n{'='*80}")
        print(f"Processing: {source_name}")
        print(f"{'='*80}")
        
        # Create output directory for this source
        output_dir = self.era2_output_path / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all files (HTML and PDF)
        html_files = list(source_dir.glob("*.html"))
        pdf_files = list(source_dir.glob("*.pdf"))
        
        all_files = html_files + pdf_files
        
        print(f"Found {len(all_files)} files ({len(html_files)} HTML, {len(pdf_files)} PDF)")
        
        if not all_files:
            print(f"  ⚠ No files to process in {source_name}")
            return
        
        # Initialize statistics for this source
        self.stats['by_source'][source_name] = {
            'total': len(all_files),
            'successful': 0,
            'failed': 0,
            'html': len(html_files),
            'pdf': len(pdf_files)
        }
        
        # Process each file
        documents = []
        file_counter = 1
        
        for file_path in all_files:
            self.stats['total_files'] += 1
            
            print(f"\n  [{file_counter}/{len(all_files)}] Processing: {file_path.name}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.html':
                result = extract_html_text(file_path)
            elif file_path.suffix.lower() == '.pdf':
                result = extract_pdf_text(file_path)
            else:
                print(f"    ⚠ Skipping unknown file type: {file_path.suffix}")
                continue
            
            if result['success'] and result.get('formatted_text'):
                # Create text file
                text_filename = self._get_text_filename(source_name, file_counter)
                text_path = output_dir / text_filename
                
                # Write formatted text (with date as first line)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result['formatted_text'])
                
                # Create document metadata entry
                doc_metadata = self._create_document_metadata(
                    result, file_path, text_filename, source_name, file_counter
                )
                documents.append(doc_metadata)
                
                self.stats['successful'] += 1
                self.stats['by_source'][source_name]['successful'] += 1
                
                print(f"    ✓ Created: {text_filename}")
                print(f"      Date: {result.get('date', 'Unknown')}")
                print(f"      Title: {result.get('title', 'N/A')[:60]}...")
                
                file_counter += 1
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"    ✗ Failed: {error_msg}")
                
                self.stats['failed'] += 1
                self.stats['by_source'][source_name]['failed'] += 1
                
                self.errors.append({
                    'source': source_name,
                    'file': file_path.name,
                    'error': error_msg
                })
        
        # Create JSON metadata file
        if documents:
            self._create_json_metadata(output_dir, source_name, documents)
            print(f"\n  ✓ Created JSON metadata file with {len(documents)} documents")
    
    def _get_text_filename(self, source_name, counter):
        """Generate text filename (e.g., infoworld_01.txt)."""
        # Clean source name (lowercase, replace spaces)
        clean_name = source_name.lower().replace(' ', '_').replace('-', '_')
        return f"{clean_name}_{counter:02d}.txt"
    
    def _create_document_metadata(self, result, original_file, text_filename, source_name, counter):
        """Create metadata entry for a document."""
        # Generate publication ID
        pub_id = f"{source_name.lower().replace(' ', '_')}_{counter:02d}"
        if result.get('date'):
            pub_id = f"{source_name.lower().replace(' ', '_')}_{result['date'].replace('-', '')}"
        
        metadata = {
            'publication_id': pub_id,
            'publication_date': result.get('date') or 'Unknown',
            'source': source_name,
            'title': result.get('title') or original_file.stem,
            'text_path': f"Era2/{source_name}/{text_filename}",
            'original_file': original_file.name,
            'era': '1980-2025'
        }
        
        # Add author if available
        if result.get('author'):
            metadata['author'] = result['author']
        
        # Add URL if available (for HTML sources)
        if result.get('url'):
            metadata['url'] = result['url']
        
        # Add file type
        metadata['file_type'] = original_file.suffix.lower()[1:]  # Remove the dot
        
        return metadata
    
    def _create_json_metadata(self, output_dir, source_name, documents):
        """Create JSON metadata file for a source."""
        # Sort documents by date (if available)
        documents.sort(key=lambda x: x.get('publication_date', '9999-99-99'))
        
        json_filename = f"{source_name}Cleaned.json"
        json_path = output_dir / json_filename
        
        # Create metadata structure
        metadata = {
            'source': source_name,
            'era': '1980-2025',
            'processed_date': datetime.now().strftime('%Y-%m-%d'),
            'total_documents': len(documents),
            'publications': documents
        }
        
        # Write JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"  Successful: {self.stats['successful']} ({self.stats['successful']/max(1,self.stats['total_files'])*100:.1f}%)")
        print(f"  Failed: {self.stats['failed']} ({self.stats['failed']/max(1,self.stats['total_files'])*100:.1f}%)")
        
        print("\nBy Source:")
        for source, stats in self.stats['by_source'].items():
            print(f"\n  {source}:")
            print(f"    Total: {stats['total']} (HTML: {stats['html']}, PDF: {stats['pdf']})")
            print(f"    Successful: {stats['successful']}")
            print(f"    Failed: {stats['failed']}")
        
        print("\n" + "="*80)
    
    def save_error_log(self):
        """Save error log to file."""
        error_log_path = self.era2_output_path / "processing_errors.log"
        
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write("ERA_2 Processing Errors\n")
            f.write("="*80 + "\n\n")
            
            for error in self.errors:
                f.write(f"Source: {error['source']}\n")
                f.write(f"File: {error['file']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write("-"*80 + "\n\n")
        
        print(f"\n⚠ Error log saved to: {error_log_path}")


def main():
    """Main execution function."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    era2_source = project_root / "Sources" / "Papers" / "ERA_2"
    era2_output = project_root / "Sources" / "CleanedSources" / "Era2"
    
    # Verify source directory exists
    if not era2_source.exists():
        print(f"Error: ERA_2 source directory not found: {era2_source}")
        return
    
    # Create processor and run
    processor = Era2Processor(era2_source, era2_output)
    processor.process_all_sources()
    
    print("\n✓ Processing complete!")
    print(f"Output directory: {era2_output}")


if __name__ == '__main__':
    main()

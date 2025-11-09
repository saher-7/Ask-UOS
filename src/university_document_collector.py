import os
import requests
import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import logging
from pathlib import Path

class UniversityDocumentCollector:
    def __init__(self, base_folder: str = "data"):
        self.base_folder = base_folder
        self.pdf_folder = os.path.join(base_folder, "pdfs")
        self.web_folder = os.path.join(base_folder, "web_content")
        
        # Create directories
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.web_folder, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # University of Sargodha specific URLs
        self.base_urls = [
            "https://su.edu.pk/Rules-and-Regulations",
            "https://su.edu.pk/admissions",
            "https://su.edu.pk/academics"
        ]
        
        # Known direct PDF links (you'll add more as you find them)
        self.known_pdfs = [
            "https://su.edu.pk/upload/rti/MS MSc Hons MPhil & PhD Rules - 2020.pdf"
        ]
        
        # Session for consistent requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_pdf(self, url: str, custom_name: str = None) -> bool:
        """Download PDF from URL with error handling"""
        try:
            self.logger.info(f" Downloading PDF: {url}")
            
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                self.logger.warning(f"  URL may not be PDF: {url}")
            
            # Determine filename
            if custom_name:
                filename = f"{custom_name}.pdf"
            else:
                filename = os.path.basename(urlparse(url).path)
                if not filename or not filename.endswith('.pdf'):
                    filename = f"document_{abs(hash(url))}.pdf"
            
            # Ensure filename is safe
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            filepath = os.path.join(self.pdf_folder, filename)
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r Progress: {percent:.1f}%", end="")
            
            print()  # New line after progress
            self.logger.info(f" Downloaded: {filename} ({downloaded} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f" Network error downloading {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f" Error downloading {url}: {e}")
            return False
    
    def scrape_page_content(self, url: str, custom_name: str = None) -> bool:
        """Scrape and save web page content"""
        try:
            self.logger.info(f" Scraping: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.page-content', 'body'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup
            
            # Extract text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean text
            text = self.clean_web_text(text)
            
            if len(text) < 200:  # Skip pages with little content
                self.logger.warning(f"  Page has minimal content: {url}")
                return False
            
            # Save content
            if custom_name:
                filename = f"{custom_name}.txt"
            else:
                page_name = url.split('/')[-1] or 'homepage'
                filename = f"web_{page_name}.txt"
            
            # Clean filename
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            filepath = os.path.join(self.web_folder, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source URL: {url}\n")
                f.write(f"Collection Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(text)
            
            self.logger.info(f" Scraped: {filename} ({len(text)} chars)")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f" Network error scraping {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f" Error scraping {url}: {e}")
            return False
    
    def clean_web_text(self, text: str) -> str:
        """Clean scraped web text"""
        import re
        
        # Split into lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty or very short lines
            if len(line) < 3:
                continue
            
            # Skip navigation and common web elements
            skip_patterns = [
                'click here', 'home', 'about us', 'contact', 'menu', 'login',
                'copyright', '©', 'all rights reserved', 'privacy policy',
                'terms of service', 'sitemap', 'search', 'follow us'
            ]
            
            if any(pattern in line.lower() for pattern in skip_patterns):
                continue
            
            # Skip lines that are mostly numbers or symbols
            if re.match(r'^[\\d\s\-\|\.]+$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Join lines and clean up spacing
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single
        
        return text.strip()
    
    def find_pdf_links(self, url: str) -> List[str]:
        """Find PDF links on a webpage"""
        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_links = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith('http'):
                    href = urljoin(url, href)
                
                # Check if it's a PDF
                if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                    pdf_links.append(href)
            
            self.logger.info(f" Found {len(pdf_links)} PDF links on {url}")
            return pdf_links
            
        except Exception as e:
            self.logger.error(f" Error finding PDFs on {url}: {e}")
            return []
    
    def collect_all_documents(self) -> Dict[str, int]:
        """Main method to collect all university documents"""
        self.logger.info(" Starting University of Sargodha document collection...")
        
        stats = {'pdfs_downloaded': 0, 'pages_scraped': 0, 'pdfs_found': 0}
        
        # Step 1: Download known PDFs
        self.logger.info("\n Phase 1: Downloading known PDF documents...")
        for pdf_url in self.known_pdfs:
            if self.download_pdf(pdf_url):
                stats['pdfs_downloaded'] += 1
            time.sleep(1)  # Be respectful
        
        # Step 2: Scrape main pages and find more PDFs
        self.logger.info("\n Phase 2: Scraping web pages and finding PDFs...")
        all_found_pdfs = set()
        
        for base_url in self.base_urls:
            # Scrape the page content
            page_name = base_url.split('/')[-1] or 'main'
            if self.scrape_page_content(base_url, f"rules_{page_name}"):
                stats['pages_scraped'] += 1
            
            # Find PDF links on this page
            found_pdfs = self.find_pdf_links(base_url)
            all_found_pdfs.update(found_pdfs)
            
            time.sleep(2)  # Be respectful
        
        # Step 3: Download newly found PDFs
        self.logger.info(f"\n Phase 3: Downloading {len(all_found_pdfs)} discovered PDFs...")
        for pdf_url in all_found_pdfs:
            if pdf_url not in self.known_pdfs:  # Skip already downloaded
                if self.download_pdf(pdf_url):
                    stats['pdfs_downloaded'] += 1
                stats['pdfs_found'] += 1
                time.sleep(1)
        
        # Step 4: Manual collection reminder
        self.logger.info("\n Phase 4: Manual collection reminder")
        print("\n" + "="*60)
        print(" MANUAL COLLECTION CHECKLIST")
        print("="*60)
        print("Please manually check and download these if available:")
        print("1. Undergraduate Rules & Regulations PDF")
        print("2. Postgraduate Rules & Regulations PDF") 
        print("3. PhD Rules & Regulations PDF")
        print("4. Examination Rules PDF")
        print("5. Admission Procedures PDF")
        print("6. Fee Structure PDF")
        print("7. Hostel Regulations PDF")
        print("8. Library Rules PDF")
        print("9. Disciplinary Guidelines PDF")
        print("10. Academic Calendar PDF")
        print("\nSave any additional PDFs to:", self.pdf_folder)
        print("Save any additional text files to:", self.web_folder)
        print("="*60)
        
        self.logger.info(f" Collection completed! Stats: {stats}")
        return stats
    
    def get_collection_summary(self) -> Dict:
        """Get summary of collected documents"""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        web_files = [f for f in os.listdir(self.web_folder) if f.endswith('.txt')]
        
        total_pdf_size = sum(
            os.path.getsize(os.path.join(self.pdf_folder, f)) 
            for f in pdf_files
        )
        
        return {
            'pdf_count': len(pdf_files),
            'web_count': len(web_files),
            'pdf_files': pdf_files,
            'web_files': web_files,
            'total_pdf_size_mb': round(total_pdf_size / (1024*1024), 2),
            'folders': {
                'pdfs': self.pdf_folder,
                'web_content': self.web_folder
            }
            
        }
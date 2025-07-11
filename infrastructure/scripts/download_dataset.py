#!/usr/bin/env python3
"""
Script to download datasets from Google Drive and extract them to data/datasets folder.
Supports both direct Google Drive file downloads and shared folder downloads.
"""

import os
import sys
import requests
import zipfile
import tarfile
import subprocess
import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Class to handle dataset downloads from Google Drive"""
    
    def __init__(self, base_dir=None):
        """
        Initialize the downloader
        
        Args:
            base_dir: Base directory path (defaults to project root)
        """
        if base_dir is None:
            # Get script directory and go up to project root
            script_dir = Path(__file__).parent
            self.base_dir = script_dir.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.datasets_dir = self.base_dir / "core" / "data" / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Datasets directory: {self.datasets_dir}")
    
    def extract_google_drive_id(self, url):
        """
        Extract Google Drive file ID from various URL formats
        
        Args:
            url: Google Drive URL
            
        Returns:
            str: File ID or None if not found
        """
        # Common Google Drive URL patterns
        patterns = [
            # Direct file link: https://drive.google.com/file/d/FILE_ID/view
            r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
            # Open link: https://drive.google.com/open?id=FILE_ID
            r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
            # Download link: https://drive.google.com/uc?id=FILE_ID
            r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',
            # Folder link: https://drive.google.com/drive/folders/FOLDER_ID
            r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Try to extract from query parameters
        parsed = urlparse(url)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            if 'id' in query_params:
                return query_params['id'][0]
        
        return None
    
    def download_file(self, url, filename=None, extract=True):
        """
        Download a file from Google Drive
        
        Args:
            url: Google Drive URL
            filename: Optional custom filename (will be auto-detected if None)
            extract: Whether to extract compressed files
            
        Returns:
            str: Path to downloaded/extracted content
        """
        file_id = self.extract_google_drive_id(url)
        if not file_id:
            raise ValueError(f"Could not extract Google Drive file ID from URL: {url}")
        
        logger.info(f"Downloading Google Drive file ID: {file_id}")
        
        # Check if it's a folder URL
        is_folder = 'folders' in url
        
        if is_folder:
            # For folders, we need to try a different approach
            logger.info("Detected Google Drive folder - trying folder download")
            return self._download_folder(file_id, filename)
        
        # Determine output filename
        if filename is None:
            # Try to get filename from gdown or use generic name
            filename = f"dataset_{file_id}"
        
        # Remove file extension for directory name, add back for file
        name_without_ext = filename.replace('.zip', '').replace('.tar.gz', '').replace('.tar', '')
        output_dir = self.datasets_dir / name_without_ext
        
        # Check if already exists
        if output_dir.exists() and extract:
            logger.info(f"Dataset already exists at: {output_dir}")
            return str(output_dir)
        
        # Download to temporary location first
        temp_file = self.datasets_dir / f"temp_{filename}"
        
        try:
            logger.info(f"Downloading to: {temp_file}")
            
            # Try multiple download methods for files
            if self._download_google_drive_file(file_id, temp_file):
                logger.info("Downloaded successfully")
            else:
                raise Exception("All download methods failed")
            
            if not temp_file.exists():
                raise Exception("Download failed - file not found")
            
            file_size = temp_file.stat().st_size
            logger.info(f"Downloaded {file_size} bytes")
            
            if file_size == 0:
                raise Exception("Downloaded file is empty")
            
            # Extract if it's a compressed file and extract=True
            if extract and self._is_compressed(temp_file):
                logger.info(f"Extracting to: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if self._extract_file(temp_file, output_dir):
                    # Remove temporary file after successful extraction
                    temp_file.unlink()
                    logger.info(f"Successfully extracted dataset to: {output_dir}")
                    return str(output_dir)
                else:
                    raise Exception("Extraction failed")
            else:
                # Move file to final location
                final_file = self.datasets_dir / filename
                temp_file.rename(final_file)
                logger.info(f"Downloaded file saved as: {final_file}")
                return str(final_file)
                
        except Exception as e:
            # Clean up temporary file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _download_folder(self, folder_id, folder_name=None):
        """
        Download Google Drive folder (simplified approach)
        """
        if folder_name is None:
            folder_name = f"folder_{folder_id}"
        
        output_dir = self.datasets_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # For now, create a simple info file since Google Drive folder downloads
        # require more complex API calls or special tools
        info_file = output_dir / "README.txt"
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        info_content = f"""
Google Drive Folder Download
===========================

Folder ID: {folder_id}
Downloaded on: {timestamp}

This folder needs to be downloaded manually or with specialized tools.
You can:
1. Open the folder in Google Drive and download as ZIP
2. Use Google Drive API with proper authentication
3. Use tools like rclone with Google Drive setup

Folder URL: https://drive.google.com/drive/folders/{folder_id}
"""
        
        with open(info_file, 'w') as f:
            f.write(info_content)
        
        logger.warning(f"Google Drive folders require manual download. Info saved to: {info_file}")
        logger.info(f"Please download manually from: https://drive.google.com/drive/folders/{folder_id}")
        
        return str(output_dir)
    
    def _download_google_drive_file(self, file_id, output_file):
        """Download Google Drive file with multiple attempts"""
        urls = [
            f"https://drive.google.com/uc?id={file_id}&export=download",
            f"https://drive.google.com/uc?id={file_id}",
            f"https://docs.google.com/uc?id={file_id}&export=download"
        ]
        
        for url in urls:
            logger.info(f"Trying URL: {url}")
            
            if self._download_with_wget(url, output_file):
                return True
            elif self._download_with_curl(url, output_file):
                return True
            elif self._download_with_requests(url, output_file):
                return True
        
        return False
    
    def _download_with_wget(self, url, output_file):
        """Download using wget"""
        try:
            cmd = [
                'wget',
                '--no-check-certificate',
                '--content-disposition',
                '-O', str(output_file),
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _download_with_curl(self, url, output_file):
        """Download using curl"""
        try:
            cmd = [
                'curl',
                '-L',  # Follow redirects
                '-o', str(output_file),
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _download_with_requests(self, url, output_file):
        """Download using Python requests"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False
    
    def _is_compressed(self, file_path):
        """Check if file is compressed"""
        file_path = Path(file_path)
        compressed_extensions = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2']
        
        # Check by extension
        for ext in compressed_extensions:
            if str(file_path).lower().endswith(ext):
                return True
        
        # Check by reading file header
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
                # ZIP file magic bytes
                if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
                    return True
                # TAR file magic bytes
                if b'ustar' in header or header.startswith(b'\x1f\x8b'):  # gzip
                    return True
        except:
            pass
        
        return False
    
    def _extract_file(self, archive_path, extract_dir):
        """Extract compressed file"""
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        
        try:
            if str(archive_path).lower().endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    logger.info(f"Extracted ZIP file with {len(zip_ref.namelist())} files")
                    return True
                    
            elif any(str(archive_path).lower().endswith(ext) for ext in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2']):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    logger.info(f"Extracted TAR file with {len(tar_ref.getnames())} files")
                    return True
            else:
                logger.warning(f"Unknown archive format: {archive_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {e}")
            return False
    
    def list_datasets(self):
        """List all downloaded datasets"""
        if not self.datasets_dir.exists():
            logger.info("No datasets directory found")
            return []
        
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                # Count files in dataset
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                
                datasets.append({
                    'name': item.name,
                    'path': str(item),
                    'files': file_count,
                    'size_mb': round(size / (1024 * 1024), 2)
                })
            elif item.is_file() and not item.name.startswith('temp_'):
                size = item.stat().st_size
                datasets.append({
                    'name': item.name,
                    'path': str(item),
                    'files': 1,
                    'size_mb': round(size / (1024 * 1024), 2)
                })
        
        return datasets

def main():
    parser = argparse.ArgumentParser(description='Download datasets from Google Drive')
    parser.add_argument('url', nargs='?', help='Google Drive URL to download')
    parser.add_argument('--name', help='Custom name for the dataset')
    parser.add_argument('--no-extract', action='store_true', help='Do not extract compressed files')
    parser.add_argument('--list', action='store_true', help='List existing datasets')
    parser.add_argument('--base-dir', help='Base directory path (defaults to project root)')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_dir)
    
    if args.list:
        datasets = downloader.list_datasets()
        if datasets:
            logger.info("Existing datasets:")
            for ds in datasets:
                logger.info(f"  {ds['name']}: {ds['files']} files, {ds['size_mb']} MB")
        else:
            logger.info("No datasets found")
        return
    
    if not args.url:
        parser.error("URL is required unless using --list")
    
    try:
        # Download the dataset
        result_path = downloader.download_file(
            args.url, 
            filename=args.name,
            extract=not args.no_extract
        )
        
        logger.info(f"✅ Dataset successfully downloaded to: {result_path}")
        
        # Show some info about the downloaded content
        result_path = Path(result_path)
        if result_path.is_dir():
            file_count = sum(1 for _ in result_path.rglob('*') if _.is_file())
            logger.info(f"📁 Contains {file_count} files")
            
            # Show first few files as example
            files = list(result_path.rglob('*'))[:10]
            if files:
                logger.info("📄 Sample files:")
                for f in files:
                    if f.is_file():
                        rel_path = f.relative_to(result_path)
                        logger.info(f"  {rel_path}")
        else:
            size = result_path.stat().st_size
            logger.info(f"📄 File size: {round(size / (1024 * 1024), 2)} MB")
            
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

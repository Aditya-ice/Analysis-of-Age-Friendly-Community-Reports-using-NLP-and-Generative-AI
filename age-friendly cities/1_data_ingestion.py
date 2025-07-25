import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Configuration ---
# The main URL to scrape for links to age-friendly communities
BASE_URL = "https://www.aarp.org/livable-communities/network-age-friendly-communities/members.html"
BASE_URL = "https://livablemap.aarp.org/member/new-york-ny"
# The directory to save downloaded PDF files
OUTPUT_DIR = "reports"

# --- Main Script ---

def download_pdfs_from_url(url, output_dir):
    """
    Finds and downloads all PDF files linked from the given URL.
    """
    print(f"🔎 Scraping URL: {url}")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    try:
        # Fetch the webpage content
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links (<a> tags) that point to a .pdf file
        pdf_links = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))
        
        if not pdf_links:
            print("❌ No PDF links found on the main page.")
            return

        print(f"✅ Found {len(pdf_links)} PDF link(s). Starting download...")
        
        for link in pdf_links:
            pdf_url = link['href']
            
            # Make sure the URL is absolute
            if not pdf_url.startswith('http'):
                pdf_url = urljoin(url, pdf_url)
                
            # Get the filename from the URL
            file_name = os.path.join(output_dir, pdf_url.split('/')[-1])
            
            # Check if the file already exists to avoid re-downloading
            if os.path.exists(file_name):
                print(f"☑️ Skipping '{file_name}', already exists.")
                continue

            try:
                # Download the PDF
                print(f"  Downloading '{file_name}' from {pdf_url}...")
                pdf_response = requests.get(pdf_url, stream=True)
                pdf_response.raise_for_status()
                
                # Save the PDF to the local directory
                with open(file_name, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  ✔️ Successfully saved '{file_name}'")

            except requests.exceptions.RequestException as e:
                print(f"  ⚠️  Could not download {pdf_url}. Error: {e}")

    except requests.exceptions.RequestException as e:
        print(f"🔥 Failed to fetch the main URL {url}. Error: {e}")

if __name__ == "__main__":
    download_pdfs_from_url(BASE_URL, OUTPUT_DIR)
    print("\n🎉 Data ingestion complete.")
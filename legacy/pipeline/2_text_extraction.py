import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- Configuration ---
INPUT_DIR = "reports"
OUTPUT_DIR = "text_output"
# For Windows users, you might need to set the Tesseract path explicitly:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a single PDF and saves it to a .txt file.
    It first tries direct text extraction, then falls back to OCR if needed.
    """
    file_name = os.path.basename(pdf_path)
    output_filename = os.path.join(output_dir, file_name.replace('.pdf', '.txt'))
    
    # Skip if the text file already exists
    if os.path.exists(output_filename):
        print(f"☑️ Skipping '{file_name}', text file already exists.")
        return

    print(f"📄 Processing '{file_name}'...")
    full_text = ""

    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # --- Method 1: Direct Text Extraction (Fast and Accurate) ---
        for page_num, page in enumerate(doc):
            full_text += page.get_text()

        # If direct extraction yields very little text, it might be an image-based PDF
        # We'll use OCR as a fallback. A threshold of 100 characters per page is a reasonable check.
        if len(full_text.strip()) < 100 * len(doc):
            print("  ⚠️ Low text yield. Trying OCR as a fallback...")
            full_text = "" # Reset text to fill with OCR content
            # --- Method 2: OCR with Tesseract (Slower but necessary for scanned docs) ---
            for page_num, page in enumerate(doc):
                # Convert page to an image
                pix = page.get_pixmap(dpi=300) # Higher DPI for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Use Tesseract to extract text from the image
                page_text = pytesseract.image_to_string(img, lang='eng')
                full_text += page_text + "\n"
        
        # Save the extracted text to a file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"  ✔️ Successfully extracted text to '{output_filename}'")
        doc.close()

    except Exception as e:
        print(f"  🔥 Failed to process '{file_name}'. Error: {e}")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: {OUTPUT_DIR}")
        
    # Process all PDF files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(INPUT_DIR, filename)
            extract_text_from_pdf(pdf_path, OUTPUT_DIR)
            
    print("\n🎉 Text extraction complete.")
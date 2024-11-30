import PyPDF2
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import os

def pdf_to_text(pdf_path):
    text = ""
    
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text from PDF content
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"Number of pages: {len(reader.pages)}")
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"--- Page {i+1} Text ---\n{page_text}\n\n"
                print(f"Extracted text from page {i+1}")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    # Convert PDF to images
    try:
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        print(f"Converted {len(images)} pages to images")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return text
    
    # Extract text from images using OCR
    for i, image in enumerate(images):
        print(f"Processing image {i+1}")
        try:
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess the image
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            config = r'--oem 3 --psm 6'
            img_text = pytesseract.image_to_string(thresh, config=config)
            text += f"\n--- Image {i+1} OCR Text ---\n{img_text}\n"
            print(f"Extracted text from image {i+1}")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
    
    return text

# Usage
pdf_path = "SaleAgreement (1).pdf"
if not os.path.exists(pdf_path):
    print(f"Error: File {pdf_path} not found")
else:
    extracted_text = pdf_to_text(pdf_path)

    # Write the extracted text to a file
    output_file = "extracted_text.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(extracted_text)

    print(f"Text extracted and written to {output_file}")
    print(f"First 500 characters of extracted text:\n{extracted_text[:500]}")

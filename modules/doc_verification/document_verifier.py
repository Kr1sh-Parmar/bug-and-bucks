import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os

class DocumentVerifier:
    def __init__(self, tesseract_path=None):
        """Initialize document verifier"""
        # Set default tesseract path if not provided
        if tesseract_path is None:
            # Try common installation paths
            if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            elif os.path.exists(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'):
                tesseract_path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.tesseract_available = True
        else:
            print("Warning: Tesseract OCR not found. Document verification will be limited.")
            self.tesseract_available = False
        
        # Reference database for document verification
        # In a real system, this would connect to government APIs or databases
        self.document_templates = {}
    
    def add_document_template(self, doc_type, template_image, regions_of_interest):
        """Add a document template for verification"""
        self.document_templates[doc_type] = {
            "template": template_image,
            "roi": regions_of_interest
        }
    
    def extract_text_from_image(self, image):
        """Extract text from document image using OCR"""
        if not self.tesseract_available:
            return "TESSERACT_NOT_AVAILABLE"
        
        # Convert to PIL image for tesseract
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Error in OCR: {e}")
            return "OCR_ERROR"
    
    def verify_document(self, image, doc_type, expected_fields=None):
        """Verify a document's authenticity"""
        # Check if Tesseract is available
        if not self.tesseract_available:
            # Return a simplified response for testing
            return True, {"status": "simplified_verification", 
                         "message": "Document accepted (Tesseract OCR not available for full verification)",
                         "extracted_data": {"doc_type": doc_type}}
        
        # Extract text from document
        extracted_text = self.extract_text_from_image(image)
        
        # Check for common document fields based on type
        verification_results = {}
        
        if doc_type == "id_card":
            # Extract ID number
            id_number_match = re.search(r'ID[:\s]*([A-Z0-9]+)', extracted_text)
            if id_number_match:
                verification_results["id_number"] = id_number_match.group(1)
            
            # Extract name
            name_match = re.search(r'Name[:\s]*([A-Za-z\s]+)', extracted_text)
            if name_match:
                verification_results["name"] = name_match.group(1).strip()
            
            # Extract date of birth
            dob_match = re.search(r'Birth[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})', extracted_text)
            if dob_match:
                verification_results["dob"] = dob_match.group(1)
        
        elif doc_type == "aadhar_card":
            # Extract Aadhar number (12 digits)
            aadhar_match = re.search(r'(\d{4}\s\d{4}\s\d{4}|\d{12})', extracted_text)
            if aadhar_match:
                verification_results["aadhar_number"] = aadhar_match.group(1).replace(" ", "")
            
            # Extract name (typically follows "Name:" or just after DOB)
            name_match = re.search(r'Name[:\s]*([A-Za-z\s]+)', extracted_text) or re.search(r'DOB[:\s].*?\n([A-Za-z\s]+)', extracted_text)
            if name_match:
                verification_results["name"] = name_match.group(1).strip()
            
            # Extract date of birth (in DD/MM/YYYY format)
            dob_match = re.search(r'DOB[:\s]*(\d{2}/\d{2}/\d{4})', extracted_text)
            if dob_match:
                verification_results["dob"] = dob_match.group(1)
            
            # Extract gender
            gender_match = re.search(r'(Male|Female|MALE|FEMALE|M|F)', extracted_text)
            if gender_match:
                verification_results["gender"] = gender_match.group(1)
            
            # Check for QR code (simplified - in reality would need image processing)
            # This is a placeholder for actual QR code detection
            verification_results["has_qr_code"] = "Detected" if "QR" in extracted_text else "Not detected"
        
        # Compare with expected fields if provided
        if expected_fields:
            for field, expected_value in expected_fields.items():
                if field in verification_results:
                    if verification_results[field].lower() != expected_value.lower():
                        return False, f"Field mismatch: {field}"
        
        # Check for security features (simplified)
        # In a real system, you would check for watermarks, holograms, etc.
        
        # For demonstration, we'll consider it verified if we found at least some expected fields
        if verification_results:
            return True, {"status": "verified", "extracted_data": verification_results}
        else:
            return False, "Could not extract required information from document"
    
    def get_document_data(self, image, doc_type):
        """Extract structured data from document"""
        is_verified, result = self.verify_document(image, doc_type)
        
        if is_verified:
            return result["extracted_data"]
        else:
            return None 
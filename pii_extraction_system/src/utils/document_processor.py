"""Document processing utilities for various file formats."""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract
from PIL import Image
from docx import Document
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdf_extract_text

from core.config import settings
from core.logging_config import get_logger

# For password-protected documents
try:
    import msoffcrypto
    MSOFFCRYPTO_AVAILABLE = True
except ImportError:
    MSOFFCRYPTO_AVAILABLE = False

# For Excel files
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Configure tesseract path for macOS Homebrew installation
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles processing of various document formats."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,  # Limited support
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.txt': self._process_text,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.tiff': self._process_image,
            '.bmp': self._process_image,
        }
        
        # Configure Tesseract
        pytesseract.pytesseract.tesseract_cmd = settings.processing.tesseract_cmd
    
    def process_document(self, file_path: Union[str, Path], password: Optional[str] = None) -> Dict:
        """
        Process a document and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            password: Optional password for protected documents
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.processing.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {settings.processing.max_file_size_mb}MB")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Pass password to processing methods that support it
            if file_extension in ['.docx', '.doc', '.pdf', '.xlsx', '.xls']:
                result = self.supported_formats[file_extension](file_path, password)
            else:
                result = self.supported_formats[file_extension](file_path)
            
            result.update({
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size_mb': round(file_size_mb, 2),
                'file_type': file_extension,
                'processing_status': 'success'
            })
            
            logger.info(f"Successfully processed: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: Path, password: Optional[str] = None) -> Dict:
        """Process PDF files."""
        result = {
            'raw_text': '',
            'ocr_text': '',
            'pages': [],
            'metadata': {},
            'bounding_boxes': []
        }
        
        try:
            # Try extracting text with PyPDF2 first
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Handle password-protected PDFs
                if pdf_reader.is_encrypted:
                    if password:
                        try:
                            pdf_reader.decrypt(password)
                            logger.info(f"Successfully decrypted PDF with provided password")
                        except Exception as e:
                            logger.error(f"Failed to decrypt PDF with password: {e}")
                            raise ValueError(f"Invalid password for encrypted PDF: {file_path}")
                    else:
                        raise ValueError(f"PDF is encrypted but no password provided: {file_path}")
                
                result['metadata'] = {
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                    'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')) if pdf_reader.metadata else '',
                    'is_encrypted': pdf_reader.is_encrypted
                }
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    result['pages'].append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    result['raw_text'] += page_text + '\n'
        
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}, trying pdfminer")
            
            # Fallback to pdfminer
            try:
                result['raw_text'] = pdf_extract_text(str(file_path))
            except Exception as e2:
                logger.warning(f"pdfminer extraction failed: {e2}")
        
        # If text extraction failed or returned very little, try OCR
        if len(result['raw_text'].strip()) < 50:
            logger.info("Text extraction yielded little content, attempting OCR")
            try:
                result['ocr_text'] = self._ocr_pdf(file_path)
            except Exception as e:
                logger.error(f"OCR failed: {e}")
        
        return result
    
    def _process_docx(self, file_path: Path, password: Optional[str] = None) -> Dict:
        """Process DOCX files."""
        result = {
            'raw_text': '',
            'ocr_text': '',
            'paragraphs': [],
            'metadata': {},
            'bounding_boxes': []
        }
        
        try:
            # Handle password-protected DOCX files
            doc_to_process = file_path
            temp_file = None
            
            if password and MSOFFCRYPTO_AVAILABLE:
                try:
                    # Check if the file is encrypted
                    with open(file_path, 'rb') as f:
                        office_file = msoffcrypto.OfficeFile(f)
                        if office_file.is_encrypted():
                            # Create temporary decrypted file
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
                            office_file.load_key(password=password)
                            office_file.decrypt(temp_file)
                            temp_file.close()
                            doc_to_process = temp_file.name
                            logger.info(f"Successfully decrypted DOCX with provided password")
                except Exception as e:
                    logger.error(f"Failed to decrypt DOCX with password: {e}")
                    if temp_file:
                        temp_file.close()
                    raise ValueError(f"Invalid password for encrypted DOCX: {file_path}")
            
            # Process the document (either original or decrypted)
            doc = Document(doc_to_process)
            
            # Extract metadata
            core_props = doc.core_properties
            result['metadata'] = {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'created': str(core_props.created) if core_props.created else '',
                'modified': str(core_props.modified) if core_props.modified else '',
                'num_paragraphs': len(doc.paragraphs),
                'is_encrypted': doc_to_process != str(file_path)
            }
            
            # Extract text from paragraphs
            for para_num, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                result['paragraphs'].append({
                    'paragraph_number': para_num + 1,
                    'text': para_text,
                    'char_count': len(para_text)
                })
                result['raw_text'] += para_text + '\n'
            
            # Extract text from tables if any
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        result['raw_text'] += cell.text + ' '
                result['raw_text'] += '\n'
            
            # Clean up temporary file if created
            if temp_file and doc_to_process != str(file_path):
                try:
                    import os
                    os.unlink(temp_file.name)
                except Exception as cleanup_e:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_e}")
        
        except Exception as e:
            # Clean up temporary file on error
            if temp_file and doc_to_process != str(file_path):
                try:
                    import os
                    os.unlink(temp_file.name)
                except Exception as cleanup_e:
                    logger.warning(f"Failed to cleanup temporary file on error: {cleanup_e}")
            
            logger.error(f"DOCX processing failed: {e}")
            raise
        
        return result
    
    def _process_excel(self, file_path: Path, password: Optional[str] = None) -> Dict:
        """Process Excel files (.xlsx, .xls)."""
        result = {
            'raw_text': '',
            'sheets': [],
            'metadata': {},
            'bounding_boxes': []
        }
        
        if not OPENPYXL_AVAILABLE:
            raise ValueError("openpyxl not available. Cannot process Excel files.")
        
        try:
            # Handle password-protected Excel files
            workbook_to_process = file_path
            temp_file = None
            
            if password and MSOFFCRYPTO_AVAILABLE:
                try:
                    # Check if the file is encrypted
                    with open(file_path, 'rb') as f:
                        office_file = msoffcrypto.OfficeFile(f)
                        if office_file.is_encrypted():
                            # Create temporary decrypted file
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                            office_file.load_key(password=password)
                            office_file.decrypt(temp_file)
                            temp_file.close()
                            workbook_to_process = temp_file.name
                            logger.info(f"Successfully decrypted Excel with provided password")
                except Exception as e:
                    logger.error(f"Failed to decrypt Excel with password: {e}")
                    if temp_file:
                        temp_file.close()
                    raise ValueError(f"Invalid password for encrypted Excel: {file_path}")
            
            # Process the Excel file (either original or decrypted)
            workbook = openpyxl.load_workbook(workbook_to_process, data_only=True)
            
            # Extract metadata
            result['metadata'] = {
                'sheet_names': workbook.sheetnames,
                'num_sheets': len(workbook.sheetnames),
                'is_encrypted': workbook_to_process != str(file_path)
            }
            
            # Process each sheet
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_text = ''
                rows_data = []
                
                # Extract data from cells
                for row_num, row in enumerate(sheet.iter_rows(values_only=True), 1):
                    row_text_parts = []
                    for cell_value in row:
                        if cell_value is not None:
                            cell_text = str(cell_value)
                            row_text_parts.append(cell_text)
                            sheet_text += cell_text + ' '
                    
                    if row_text_parts:  # Only add non-empty rows
                        rows_data.append({
                            'row_number': row_num,
                            'cells': row_text_parts,
                            'text': ' '.join(row_text_parts)
                        })
                    
                    sheet_text += '\n'
                
                result['sheets'].append({
                    'sheet_name': sheet_name,
                    'text': sheet_text,
                    'rows': rows_data,
                    'num_rows': len(rows_data)
                })
                
                result['raw_text'] += f"Sheet: {sheet_name}\n{sheet_text}\n"
            
            # Clean up temporary file if created
            if temp_file and workbook_to_process != str(file_path):
                try:
                    import os
                    os.unlink(temp_file.name)
                except Exception as cleanup_e:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_e}")
        
        except Exception as e:
            # Clean up temporary file on error
            if temp_file and workbook_to_process != str(file_path):
                try:
                    import os
                    os.unlink(temp_file.name)
                except Exception as cleanup_e:
                    logger.warning(f"Failed to cleanup temporary file on error: {cleanup_e}")
            
            logger.error(f"Excel processing failed: {e}")
            raise
        
        return result
    
    def _process_text(self, file_path: Path) -> Dict:
        """Process plain text files."""
        result = {
            'raw_text': '',
            'ocr_text': '',
            'lines': [],
            'metadata': {},
            'bounding_boxes': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                result['raw_text'] = content
                
                lines = content.split('\n')
                result['lines'] = [{'line_number': i+1, 'text': line} for i, line in enumerate(lines)]
                result['metadata'] = {
                    'num_lines': len(lines),
                    'char_count': len(content),
                    'encoding': 'utf-8'
                }
        
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            raise
        
        return result
    
    def _process_image(self, file_path: Path) -> Dict:
        """Process image files with OCR."""
        result = {
            'raw_text': '',
            'ocr_text': '',
            'metadata': {},
            'bounding_boxes': []
        }
        
        try:
            # Load and enhance image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Get image metadata
            height, width = image.shape[:2]
            result['metadata'] = {
                'width': width,
                'height': height,
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'file_size_kb': round(file_path.stat().st_size / 1024, 2)
            }
            
            # Enhance image for better OCR
            enhanced_image = self._enhance_image_for_ocr(image)
            
            # Perform OCR
            pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
            
            # Extract text with bounding box information
            ocr_data = pytesseract.image_to_data(
                pil_image,
                lang=settings.processing.ocr_languages,
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            text_parts = []
            bounding_boxes = []
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 30:  # Filter low-confidence results
                    text_parts.append(text)
                    
                    # Store bounding box information
                    bounding_boxes.append({
                        'text': text,
                        'confidence': confidence,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    })
            
            result['ocr_text'] = ' '.join(text_parts)
            result['raw_text'] = result['ocr_text']  # For images, OCR is the raw text
            result['bounding_boxes'] = bounding_boxes
            
            logger.info(f"OCR extracted {len(text_parts)} text elements from {file_path}")
        
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
        
        return result
    
    def _enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR results."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def _ocr_pdf(self, file_path: Path) -> str:
        """Perform OCR on PDF pages."""
        try:
            # This would require pdf2image library for proper implementation
            # For now, return empty string as placeholder
            logger.warning("PDF OCR not fully implemented - requires pdf2image")
            return ""
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return ""
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys())
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
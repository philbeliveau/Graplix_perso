"""
Format Handlers Module

Specialized handlers for different document formats, building on existing document processing
capabilities with enhanced features for data pipeline operations.
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

# Import existing components
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.logging_config import get_logger
from utils.document_processor import DocumentProcessor

logger = get_logger(__name__)

@dataclass
class ProcessingResult:
    """Enhanced processing result with pipeline-specific features."""
    
    success: bool
    document_id: str
    file_path: str
    processing_time: float
    
    # Content extraction
    raw_text: str = ""
    structured_data: Dict = None
    metadata: Dict = None
    
    # Quality metrics
    confidence_score: float = 0.0
    quality_metrics: Dict = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    # Processing details
    processing_method: str = ""
    preprocessing_applied: List[str] = None
    
    def __post_init__(self):
        if self.structured_data is None:
            self.structured_data = {}
        if self.metadata is None:
            self.metadata = {}
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.warnings is None:
            self.warnings = []
        if self.preprocessing_applied is None:
            self.preprocessing_applied = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'document_id': self.document_id,
            'file_path': self.file_path,
            'processing_time': self.processing_time,
            'raw_text': self.raw_text,
            'structured_data': self.structured_data,
            'metadata': self.metadata,
            'confidence_score': self.confidence_score,
            'quality_metrics': self.quality_metrics,
            'error_message': self.error_message,
            'warnings': self.warnings,
            'processing_method': self.processing_method,
            'preprocessing_applied': self.preprocessing_applied,
            'timestamp': datetime.now().isoformat()
        }


class FormatHandler(ABC):
    """Abstract base class for format-specific handlers."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        """Initialize handler with document processor."""
        self.document_processor = document_processor or DocumentProcessor()
        self.supported_extensions = []
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the given file."""
        pass
    
    @abstractmethod
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process the document and return enhanced results."""
        pass
    
    def get_format_info(self) -> Dict:
        """Get information about this format handler."""
        return {
            'handler_name': self.__class__.__name__,
            'supported_extensions': self.supported_extensions,
            'capabilities': self._get_capabilities()
        }
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this handler."""
        return ['text_extraction', 'metadata_extraction']
    
    def _calculate_quality_metrics(self, result: Dict, processing_time: float) -> Dict:
        """Calculate quality metrics for the processing result."""
        metrics = {}
        
        # Text quality metrics
        text = result.get('raw_text', '')
        if text:
            metrics['text_length'] = len(text)
            metrics['word_count'] = len(text.split())
            metrics['line_count'] = len(text.split('\n'))
            metrics['non_empty_lines'] = len([line for line in text.split('\n') if line.strip()])
        
        # Processing performance metrics
        metrics['processing_time_seconds'] = processing_time
        metrics['processing_speed_chars_per_second'] = len(text) / processing_time if processing_time > 0 else 0
        
        return metrics


class PDFHandler(FormatHandler):
    """Enhanced PDF processing handler."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        super().__init__(document_processor)
        self.supported_extensions = ['.pdf']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this is a PDF file."""
        return file_path.suffix.lower() == '.pdf'
    
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process PDF with enhanced features."""
        start_time = time.time()
        
        try:
            # Use existing document processor
            result = self.document_processor.process_document(file_path, kwargs.get('password'))
            
            processing_time = time.time() - start_time
            
            # Calculate confidence based on text extraction success
            confidence_score = self._calculate_pdf_confidence(result)
            
            # Extract structured data
            structured_data = self._extract_pdf_structure(result)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, processing_time)
            quality_metrics.update(self._get_pdf_specific_metrics(result))
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                raw_text=result.get('raw_text', '') or result.get('ocr_text', ''),
                structured_data=structured_data,
                metadata=result.get('metadata', {}),
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                processing_method='pdf_extraction_with_ocr_fallback',
                preprocessing_applied=self._get_pdf_preprocessing_steps(result)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDF processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                error_message=str(e),
                processing_method='pdf_extraction_failed'
            )
    
    def _calculate_pdf_confidence(self, result: Dict) -> float:
        """Calculate confidence score for PDF processing."""
        raw_text = result.get('raw_text', '')
        ocr_text = result.get('ocr_text', '')
        
        # If we have raw text, high confidence
        if raw_text and len(raw_text.strip()) > 50:
            return 0.9
        
        # If we had to use OCR, moderate confidence
        if ocr_text and len(ocr_text.strip()) > 20:
            return 0.7
        
        # If very little text extracted, low confidence
        total_text = raw_text + ocr_text
        if len(total_text.strip()) > 10:
            return 0.4
        
        return 0.1
    
    def _extract_pdf_structure(self, result: Dict) -> Dict:
        """Extract structured data from PDF processing result."""
        structured = {
            'pages': result.get('pages', []),
            'has_raw_text': bool(result.get('raw_text', '').strip()),
            'has_ocr_text': bool(result.get('ocr_text', '').strip()),
            'bounding_boxes': result.get('bounding_boxes', [])
        }
        
        # Extract metadata
        pdf_metadata = result.get('metadata', {})
        if pdf_metadata:
            structured['document_info'] = {
                'title': pdf_metadata.get('title', ''),
                'author': pdf_metadata.get('author', ''),
                'creation_date': pdf_metadata.get('creation_date', ''),
                'num_pages': pdf_metadata.get('num_pages', 0),
                'is_encrypted': pdf_metadata.get('is_encrypted', False)
            }
        
        return structured
    
    def _get_pdf_specific_metrics(self, result: Dict) -> Dict:
        """Get PDF-specific quality metrics."""
        metrics = {}
        
        metadata = result.get('metadata', {})
        if metadata:
            metrics['num_pages'] = metadata.get('num_pages', 0)
            metrics['is_encrypted'] = metadata.get('is_encrypted', False)
        
        # Text extraction method used
        has_raw = bool(result.get('raw_text', '').strip())
        has_ocr = bool(result.get('ocr_text', '').strip())
        
        if has_raw and has_ocr:
            metrics['extraction_method'] = 'hybrid'
        elif has_raw:
            metrics['extraction_method'] = 'direct'
        elif has_ocr:
            metrics['extraction_method'] = 'ocr'
        else:
            metrics['extraction_method'] = 'failed'
        
        return metrics
    
    def _get_pdf_preprocessing_steps(self, result: Dict) -> List[str]:
        """Get list of preprocessing steps applied."""
        steps = ['pdf_text_extraction']
        
        if result.get('ocr_text'):
            steps.append('ocr_fallback')
        
        if result.get('metadata', {}).get('is_encrypted'):
            steps.append('decryption')
        
        return steps
    
    def _get_capabilities(self) -> List[str]:
        """PDF handler capabilities."""
        return [
            'text_extraction', 'metadata_extraction', 'ocr_fallback',
            'page_analysis', 'structure_detection', 'encrypted_pdf_support'
        ]


class ImageHandler(FormatHandler):
    """Enhanced image processing handler with OCR capabilities."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        super().__init__(document_processor)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this is a supported image file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process image with OCR and quality analysis."""
        start_time = time.time()
        
        try:
            # Use existing document processor
            result = self.document_processor.process_document(file_path)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence based on OCR results
            confidence_score = self._calculate_image_confidence(result)
            
            # Extract structured data
            structured_data = self._extract_image_structure(result)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, processing_time)
            quality_metrics.update(self._get_image_specific_metrics(result))
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                raw_text=result.get('ocr_text', ''),
                structured_data=structured_data,
                metadata=result.get('metadata', {}),
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                processing_method=result.get('ocr_engine', 'unknown'),
                preprocessing_applied=self._get_image_preprocessing_steps(result)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                error_message=str(e),
                processing_method='image_ocr_failed'
            )
    
    def _calculate_image_confidence(self, result: Dict) -> float:
        """Calculate confidence score for image OCR."""
        ocr_text = result.get('ocr_text', '')
        bounding_boxes = result.get('bounding_boxes', [])
        
        if not ocr_text.strip():
            return 0.1
        
        # Calculate average confidence from bounding boxes if available
        if bounding_boxes:
            confidences = [box.get('confidence', 0) for box in bounding_boxes]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            return min(avg_confidence / 100.0, 1.0)  # Convert percentage to 0-1
        
        # Heuristic based on text length and content
        text_length = len(ocr_text.strip())
        if text_length > 100:
            return 0.8
        elif text_length > 50:
            return 0.6
        elif text_length > 20:
            return 0.4
        else:
            return 0.2
    
    def _extract_image_structure(self, result: Dict) -> Dict:
        """Extract structured data from image processing result."""
        structured = {
            'ocr_engine': result.get('ocr_engine', 'unknown'),
            'bounding_boxes': result.get('bounding_boxes', []),
            'image_properties': result.get('metadata', {})
        }
        
        # Group bounding boxes by confidence levels
        bboxes = result.get('bounding_boxes', [])
        if bboxes:
            high_conf = [b for b in bboxes if b.get('confidence', 0) > 80]
            med_conf = [b for b in bboxes if 50 <= b.get('confidence', 0) <= 80]
            low_conf = [b for b in bboxes if b.get('confidence', 0) < 50]
            
            structured['confidence_analysis'] = {
                'high_confidence_detections': len(high_conf),
                'medium_confidence_detections': len(med_conf),
                'low_confidence_detections': len(low_conf)
            }
        
        return structured
    
    def _get_image_specific_metrics(self, result: Dict) -> Dict:
        """Get image-specific quality metrics."""
        metrics = {}
        
        metadata = result.get('metadata', {})
        if metadata:
            metrics['image_width'] = metadata.get('width', 0)
            metrics['image_height'] = metadata.get('height', 0)
            metrics['file_size_kb'] = metadata.get('file_size_kb', 0)
            metrics['is_jpg'] = metadata.get('is_jpg', False)
            metrics['compression_ratio'] = metadata.get('compression_ratio', 0)
        
        # OCR quality metrics
        bboxes = result.get('bounding_boxes', [])
        if bboxes:
            confidences = [box.get('confidence', 0) for box in bboxes]
            metrics['ocr_detections'] = len(bboxes)
            metrics['avg_ocr_confidence'] = sum(confidences) / len(confidences)
            metrics['min_ocr_confidence'] = min(confidences)
            metrics['max_ocr_confidence'] = max(confidences)
        
        return metrics
    
    def _get_image_preprocessing_steps(self, result: Dict) -> List[str]:
        """Get list of preprocessing steps applied."""
        steps = ['image_loading']
        
        metadata = result.get('metadata', {})
        if metadata.get('is_jpg'):
            steps.append('jpg_compression_handling')
        
        ocr_engine = result.get('ocr_engine', '')
        if 'tesseract' in ocr_engine:
            steps.append('tesseract_ocr')
        if 'easyocr' in ocr_engine:
            steps.append('easyocr')
        if 'llm' in ocr_engine:
            steps.append('llm_ocr')
        
        steps.append('image_enhancement')
        
        return steps
    
    def _get_capabilities(self) -> List[str]:
        """Image handler capabilities."""
        return [
            'ocr_text_extraction', 'multiple_ocr_engines', 'image_enhancement',
            'bounding_box_detection', 'confidence_scoring', 'format_optimization'
        ]


class ExcelHandler(FormatHandler):
    """Enhanced Excel/spreadsheet processing handler."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        super().__init__(document_processor)
        self.supported_extensions = ['.xlsx', '.xls']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this is an Excel file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process Excel file with structured data extraction."""
        start_time = time.time()
        
        try:
            # Use existing document processor
            result = self.document_processor.process_document(file_path, kwargs.get('password'))
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (Excel processing is generally reliable)
            confidence_score = 0.9 if result.get('raw_text') else 0.1
            
            # Extract structured data
            structured_data = self._extract_excel_structure(result)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, processing_time)
            quality_metrics.update(self._get_excel_specific_metrics(result))
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                raw_text=result.get('raw_text', ''),
                structured_data=structured_data,
                metadata=result.get('metadata', {}),
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                processing_method='excel_extraction',
                preprocessing_applied=self._get_excel_preprocessing_steps(result)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Excel processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                error_message=str(e),
                processing_method='excel_extraction_failed'
            )
    
    def _extract_excel_structure(self, result: Dict) -> Dict:
        """Extract structured data from Excel processing result."""
        structured = {
            'sheets': result.get('sheets', []),
            'sheet_names': [],
            'total_sheets': 0,
            'total_rows': 0,
            'has_data': False
        }
        
        sheets = result.get('sheets', [])
        if sheets:
            structured['sheet_names'] = [sheet.get('sheet_name', '') for sheet in sheets]
            structured['total_sheets'] = len(sheets)
            structured['total_rows'] = sum(sheet.get('num_rows', 0) for sheet in sheets)
            structured['has_data'] = any(sheet.get('rows', []) for sheet in sheets)
            
            # Analyze sheet contents
            structured['sheet_analysis'] = []
            for sheet in sheets:
                analysis = {
                    'name': sheet.get('sheet_name', ''),
                    'row_count': sheet.get('num_rows', 0),
                    'has_header_row': self._detect_header_row(sheet),
                    'data_types': self._analyze_data_types(sheet)
                }
                structured['sheet_analysis'].append(analysis)
        
        return structured
    
    def _detect_header_row(self, sheet: Dict) -> bool:
        """Detect if sheet has a header row."""
        rows = sheet.get('rows', [])
        if not rows:
            return False
        
        # Simple heuristic: check if first row has more text than numbers
        first_row = rows[0].get('cells', []) if rows else []
        if not first_row:
            return False
        
        text_count = sum(1 for cell in first_row if not str(cell).replace('.', '').isdigit())
        return text_count > len(first_row) / 2
    
    def _analyze_data_types(self, sheet: Dict) -> Dict:
        """Analyze data types in the sheet."""
        rows = sheet.get('rows', [])
        if not rows:
            return {}
        
        types = {'text': 0, 'numeric': 0, 'empty': 0}
        total_cells = 0
        
        for row in rows:
            for cell in row.get('cells', []):
                total_cells += 1
                cell_str = str(cell).strip()
                
                if not cell_str:
                    types['empty'] += 1
                elif cell_str.replace('.', '').replace('-', '').isdigit():
                    types['numeric'] += 1
                else:
                    types['text'] += 1
        
        # Convert to percentages
        if total_cells > 0:
            return {k: round((v / total_cells) * 100, 1) for k, v in types.items()}
        
        return types
    
    def _get_excel_specific_metrics(self, result: Dict) -> Dict:
        """Get Excel-specific quality metrics."""
        metrics = {}
        
        metadata = result.get('metadata', {})
        if metadata:
            metrics['num_sheets'] = metadata.get('num_sheets', 0)
            metrics['is_encrypted'] = metadata.get('is_encrypted', False)
        
        sheets = result.get('sheets', [])
        if sheets:
            metrics['total_rows'] = sum(sheet.get('num_rows', 0) for sheet in sheets)
            metrics['non_empty_sheets'] = sum(1 for sheet in sheets if sheet.get('rows'))
            
            # Calculate data density
            total_cells = sum(
                len(row.get('cells', [])) 
                for sheet in sheets 
                for row in sheet.get('rows', [])
            )
            metrics['total_cells'] = total_cells
        
        return metrics
    
    def _get_excel_preprocessing_steps(self, result: Dict) -> List[str]:
        """Get list of preprocessing steps applied."""
        steps = ['excel_loading']
        
        if result.get('metadata', {}).get('is_encrypted'):
            steps.append('decryption')
        
        steps.extend(['sheet_parsing', 'cell_extraction', 'data_type_detection'])
        
        return steps
    
    def _get_capabilities(self) -> List[str]:
        """Excel handler capabilities."""
        return [
            'multi_sheet_support', 'structured_data_extraction', 'data_type_analysis',
            'header_detection', 'encrypted_file_support', 'cell_level_access'
        ]


class DocHandler(FormatHandler):
    """Enhanced DOC/DOCX processing handler."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        super().__init__(document_processor)
        self.supported_extensions = ['.docx', '.doc']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this is a Word document."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process Word document with structure analysis."""
        start_time = time.time()
        
        try:
            # Use existing document processor
            result = self.document_processor.process_document(file_path, kwargs.get('password'))
            
            processing_time = time.time() - start_time
            
            # Calculate confidence
            confidence_score = 0.9 if result.get('raw_text') else 0.1
            
            # Extract structured data
            structured_data = self._extract_doc_structure(result)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, processing_time)
            quality_metrics.update(self._get_doc_specific_metrics(result))
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                raw_text=result.get('raw_text', ''),
                structured_data=structured_data,
                metadata=result.get('metadata', {}),
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                processing_method='docx_extraction',
                preprocessing_applied=self._get_doc_preprocessing_steps(result)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"DOC processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                error_message=str(e),
                processing_method='docx_extraction_failed'
            )
    
    def _extract_doc_structure(self, result: Dict) -> Dict:
        """Extract structured data from DOC processing result."""
        structured = {
            'paragraphs': result.get('paragraphs', []),
            'has_tables': False,  # Placeholder - could be enhanced
            'document_structure': self._analyze_document_structure(result)
        }
        
        return structured
    
    def _analyze_document_structure(self, result: Dict) -> Dict:
        """Analyze document structure and organization."""
        paragraphs = result.get('paragraphs', [])
        
        analysis = {
            'total_paragraphs': len(paragraphs),
            'non_empty_paragraphs': 0,
            'avg_paragraph_length': 0,
            'estimated_sections': 0
        }
        
        if paragraphs:
            non_empty = [p for p in paragraphs if p.get('text', '').strip()]
            analysis['non_empty_paragraphs'] = len(non_empty)
            
            if non_empty:
                total_chars = sum(len(p.get('text', '')) for p in non_empty)
                analysis['avg_paragraph_length'] = total_chars / len(non_empty)
        
        return analysis
    
    def _get_doc_specific_metrics(self, result: Dict) -> Dict:
        """Get DOC-specific quality metrics."""
        metrics = {}
        
        metadata = result.get('metadata', {})
        if metadata:
            metrics['num_paragraphs'] = metadata.get('num_paragraphs', 0)
            metrics['is_encrypted'] = metadata.get('is_encrypted', False)
            metrics['title'] = metadata.get('title', '')
            metrics['author'] = metadata.get('author', '')
        
        return metrics
    
    def _get_doc_preprocessing_steps(self, result: Dict) -> List[str]:
        """Get list of preprocessing steps applied."""
        steps = ['docx_loading']
        
        if result.get('metadata', {}).get('is_encrypted'):
            steps.append('decryption')
        
        steps.extend(['paragraph_extraction', 'table_extraction', 'metadata_extraction'])
        
        return steps
    
    def _get_capabilities(self) -> List[str]:
        """DOC handler capabilities."""
        return [
            'text_extraction', 'paragraph_structure', 'table_extraction',
            'metadata_extraction', 'encrypted_file_support', 'legacy_doc_warning'
        ]


class TextHandler(FormatHandler):
    """Simple text file processing handler."""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        super().__init__(document_processor)
        self.supported_extensions = ['.txt']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this is a text file."""
        return file_path.suffix.lower() == '.txt'
    
    def process(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process text file with encoding detection and analysis."""
        start_time = time.time()
        
        try:
            # Use existing document processor
            result = self.document_processor.process_document(file_path)
            
            processing_time = time.time() - start_time
            
            # Text files have high confidence if successfully loaded
            confidence_score = 0.95 if result.get('raw_text') else 0.1
            
            # Extract structured data
            structured_data = self._extract_text_structure(result)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, processing_time)
            quality_metrics.update(self._get_text_specific_metrics(result))
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                raw_text=result.get('raw_text', ''),
                structured_data=structured_data,
                metadata=result.get('metadata', {}),
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                processing_method='text_file_reading',
                preprocessing_applied=['encoding_detection', 'line_parsing']
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Text processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=processing_time,
                error_message=str(e),
                processing_method='text_file_reading_failed'
            )
    
    def _extract_text_structure(self, result: Dict) -> Dict:
        """Extract structured data from text processing result."""
        lines = result.get('lines', [])
        
        structured = {
            'lines': lines,
            'line_analysis': self._analyze_text_lines(lines),
            'content_type': self._detect_content_type(result.get('raw_text', ''))
        }
        
        return structured
    
    def _analyze_text_lines(self, lines: List[Dict]) -> Dict:
        """Analyze text line structure."""
        if not lines:
            return {}
        
        analysis = {
            'total_lines': len(lines),
            'empty_lines': sum(1 for line in lines if not line.get('text', '').strip()),
            'avg_line_length': 0,
            'max_line_length': 0
        }
        
        non_empty_lines = [line for line in lines if line.get('text', '').strip()]
        if non_empty_lines:
            lengths = [len(line.get('text', '')) for line in non_empty_lines]
            analysis['avg_line_length'] = sum(lengths) / len(lengths)
            analysis['max_line_length'] = max(lengths)
        
        return analysis
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in the text file."""
        if not text:
            return 'empty'
        
        # Simple heuristics
        if text.startswith('<?xml') or text.startswith('<'):
            return 'xml/html'
        elif text.startswith('{') or text.startswith('['):
            return 'json'
        elif ',' in text and len(text.split('\n')) > 1:
            return 'csv-like'
        elif any(keyword in text.lower() for keyword in ['dear', 'sincerely', 'regards']):
            return 'letter/email'
        else:
            return 'general_text'
    
    def _get_text_specific_metrics(self, result: Dict) -> Dict:
        """Get text-specific quality metrics."""
        metrics = {}
        
        metadata = result.get('metadata', {})
        if metadata:
            metrics['num_lines'] = metadata.get('num_lines', 0)
            metrics['char_count'] = metadata.get('char_count', 0)
            metrics['word_count'] = metadata.get('word_count', 0)
            metrics['encoding'] = metadata.get('encoding', 'unknown')
        
        return metrics
    
    def _get_capabilities(self) -> List[str]:
        """Text handler capabilities."""
        return [
            'encoding_detection', 'line_analysis', 'content_type_detection',
            'structure_analysis', 'fast_processing'
        ]


class FormatHandlerRegistry:
    """Registry for managing format handlers."""
    
    def __init__(self):
        """Initialize the registry with default handlers."""
        self.handlers: List[FormatHandler] = []
        self.document_processor = DocumentProcessor()
        
        # Register default handlers
        self.register_default_handlers()
    
    def register_default_handlers(self):
        """Register all default format handlers."""
        handlers = [
            PDFHandler(self.document_processor),
            ImageHandler(self.document_processor),
            ExcelHandler(self.document_processor),
            DocHandler(self.document_processor),
            TextHandler(self.document_processor)
        ]
        
        for handler in handlers:
            self.register_handler(handler)
    
    def register_handler(self, handler: FormatHandler):
        """Register a new format handler."""
        self.handlers.append(handler)
        logger.info(f"Registered handler: {handler.__class__.__name__}")
    
    def get_handler_for_file(self, file_path: Path) -> Optional[FormatHandler]:
        """Get the appropriate handler for a file."""
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return handler
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get all supported file formats."""
        formats = set()
        for handler in self.handlers:
            formats.update(handler.supported_extensions)
        return sorted(list(formats))
    
    def get_handler_info(self) -> List[Dict]:
        """Get information about all registered handlers."""
        return [handler.get_format_info() for handler in self.handlers]
    
    def process_file(self, file_path: Path, document_id: str, **kwargs) -> ProcessingResult:
        """Process a file using the appropriate handler."""
        handler = self.get_handler_for_file(file_path)
        
        if handler is None:
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=0.0,
                error_message=f"No handler available for file type: {file_path.suffix}",
                processing_method='no_handler_available'
            )
        
        return handler.process(file_path, document_id, **kwargs)
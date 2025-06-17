"""Main PII extraction pipeline orchestrating all components."""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.config import settings
from core.logging_config import get_logger, audit_log
from utils.data_storage import storage_manager
from utils.document_processor import DocumentProcessor
from extractors.base import PIIExtractionResult, PIIEntity

logger = get_logger(__name__)


class PIIExtractionPipeline:
    """Main pipeline for PII extraction from documents."""
    
    def __init__(self, 
                 data_source: str = None,
                 models: List[str] = None,
                 config_override: Dict = None):
        """
        Initialize the PII extraction pipeline.
        
        Args:
            data_source: Override data source ('local' or 's3')
            models: List of models to use for extraction
            config_override: Configuration overrides
        """
        self.document_processor = DocumentProcessor()
        self.storage_manager = storage_manager
        
        # Override configuration if provided
        if data_source:
            settings.data_source.source_type = data_source
        
        if models:
            settings.ml_models.enabled_models = models
        
        if config_override:
            for key, value in config_override.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
        
        # Initialize extractors based on enabled models
        self.extractors = self._initialize_extractors()
        
        logger.info(f"Pipeline initialized with {len(self.extractors)} extractors")
        logger.info(f"Data source: {settings.data_source.source_type}")
        logger.info(f"Enabled models: {settings.ml_models.enabled_models}")
    
    def _initialize_extractors(self) -> Dict:
        """Initialize PII extractors based on configuration."""
        extractors = {}
        
        # Import and initialize extractors based on enabled models
        # This will be implemented as we create the extractor modules
        
        try:
            if "rule_based" in settings.ml_models.enabled_models:
                from extractors.rule_based import RuleBasedExtractor
                extractors["rule_based"] = RuleBasedExtractor()
                logger.info("Rule-based extractor initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize rule-based extractor: {e}")
        
        try:
            if "ner" in settings.ml_models.enabled_models:
                from extractors.ner_extractor import NERExtractor
                extractors["ner"] = NERExtractor()
                logger.info("NER extractor initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize NER extractor: {e}")
        
        try:
            if "layout_aware" in settings.ml_models.enabled_models:
                from extractors.layout_aware import LayoutAwareExtractor
                extractors["layout_aware"] = LayoutAwareExtractor()
                logger.info("Layout-aware extractor initialized")
        except ImportError as e:
            logger.warning(f"Failed to initialize layout-aware extractor: {e}")
        
        if not extractors:
            logger.warning("No extractors initialized - falling back to rule-based")
            from extractors.rule_based import RuleBasedExtractor
            extractors["rule_based"] = RuleBasedExtractor()
        
        return extractors
    
    def extract_from_file(self, 
                         file_path: Union[str, Path],
                         document_id: str = None,
                         save_results: bool = True) -> PIIExtractionResult:
        """
        Extract PII from a single document file.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            save_results: Whether to save extraction results
            
        Returns:
            PIIExtractionResult containing all extraction results
        """
        file_path = Path(file_path)
        
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        audit_log(f"Starting PII extraction for document: {file_path}", 
                 document_id=document_id)
        
        try:
            # Process document
            logger.info(f"Processing document: {file_path}")
            processed_doc = self.document_processor.process_document(file_path)
            
            # Extract PII using all enabled extractors
            extraction_results = {}
            for extractor_name, extractor in self.extractors.items():
                logger.info(f"Running {extractor_name} extractor")
                
                try:
                    result = extractor.extract_pii(processed_doc)
                    extraction_results[extractor_name] = result
                    
                    logger.info(f"{extractor_name} found {len(result.pii_entities)} PII entities")
                    
                except Exception as e:
                    logger.error(f"Error in {extractor_name} extractor: {e}")
                    extraction_results[extractor_name] = PIIExtractionResult(
                        pii_entities=[],
                        confidence_scores=[],
                        processing_time=0.0,
                        error=str(e)
                    )
            
            # Combine results from all extractors
            combined_result = self._combine_extraction_results(
                extraction_results, 
                processed_doc,
                document_id
            )
            
            # Save results if requested
            if save_results:
                self._save_extraction_results(document_id, combined_result, processed_doc)
            
            audit_log(f"PII extraction completed for document: {file_path}",
                     document_id=document_id,
                     pii_count=len(combined_result.pii_entities),
                     extractors_used=list(extraction_results.keys()))
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Pipeline error for {file_path}: {e}")
            audit_log(f"PII extraction failed for document: {file_path}",
                     document_id=document_id,
                     error=str(e))
            raise
    
    def extract_from_directory(self, 
                              directory_path: Union[str, Path],
                              recursive: bool = True,
                              batch_size: int = 10) -> Dict[str, PIIExtractionResult]:
        """
        Extract PII from all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            batch_size: Number of documents to process in parallel
            
        Returns:
            Dictionary mapping document IDs to extraction results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported documents
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and self.document_processor.is_supported_format(file_path):
                documents.append(file_path)
        
        logger.info(f"Found {len(documents)} supported documents in {directory_path}")
        
        # Process documents
        results = {}
        for i, doc_path in enumerate(documents):
            try:
                document_id = f"batch_{i+1:04d}_{doc_path.stem}"
                result = self.extract_from_file(doc_path, document_id)
                results[document_id] = result
                
                logger.info(f"Processed {i+1}/{len(documents)}: {doc_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
                results[f"error_{i+1:04d}_{doc_path.stem}"] = PIIExtractionResult(
                    pii_entities=[],
                    confidence_scores=[],
                    processing_time=0.0,
                    error=str(e)
                )
        
        return results
    
    def _combine_extraction_results(self, 
                                   extractor_results: Dict,
                                   processed_doc: Dict,
                                   document_id: str) -> PIIExtractionResult:
        """Combine results from multiple extractors."""
        all_entities = []
        all_confidence_scores = []
        total_processing_time = 0.0
        extractor_results_summary = {}
        
        for extractor_name, result in extractor_results.items():
            # Add extractor name to each entity
            for entity in result.pii_entities:
                # Entity already has extractor name set in the PIIEntity object
                # Just add it to the list
                all_entities.append(entity)
            
            all_confidence_scores.extend(result.confidence_scores)
            total_processing_time += result.processing_time
            
            extractor_results_summary[extractor_name] = {
                'entity_count': len(result.pii_entities),
                'processing_time': result.processing_time,
                'error': result.error
            }
        
        # Remove duplicates and merge similar entities
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        # Create combined result
        combined_result = PIIExtractionResult(
            pii_entities=deduplicated_entities,
            confidence_scores=all_confidence_scores,
            processing_time=total_processing_time,
            metadata={
                'document_id': document_id,
                'document_type': processed_doc.get('file_type', 'unknown'),
                'extractor_results': extractor_results_summary,
                'total_entities_before_dedup': len(all_entities),
                'total_entities_after_dedup': len(deduplicated_entities)
            }
        )
        
        return combined_result
    
    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove duplicate PII entities."""
        # Simple deduplication based on text and type
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.text, entity.pii_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def _save_extraction_results(self, 
                                document_id: str,
                                result: PIIExtractionResult,
                                processed_doc: Dict) -> None:
        """Save extraction results to storage."""
        try:
            # Prepare data for storage
            storage_data = {
                'document_id': document_id,
                'extraction_result': result.to_dict(),
                'processed_document': processed_doc,
                'timestamp': result.metadata.get('timestamp'),
                'settings': {
                    'data_source': settings.data_source.source_type,
                    'enabled_models': settings.ml_models.enabled_models,
                    'privacy_settings': {
                        'redaction_enabled': settings.privacy.enable_redaction,
                        'compliance_mode': {
                            'gdpr': settings.privacy.gdpr_compliance,
                            'law25': settings.privacy.law25_compliance
                        }
                    }
                }
            }
            
            # Save to storage
            success = self.storage_manager.save_processing_result(document_id, storage_data)
            
            if success:
                logger.info(f"Extraction results saved for document: {document_id}")
            else:
                logger.error(f"Failed to save extraction results for document: {document_id}")
                
        except Exception as e:
            logger.error(f"Error saving extraction results for {document_id}: {e}")
    
    def load_extraction_results(self, document_id: str) -> Optional[PIIExtractionResult]:
        """Load previously saved extraction results."""
        try:
            data = self.storage_manager.load_processing_result(document_id)
            
            if data and 'extraction_result' in data:
                return PIIExtractionResult.from_dict(data['extraction_result'])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading extraction results for {document_id}: {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return self.document_processor.get_supported_formats()
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            'data_source': settings.data_source.source_type,
            'enabled_models': settings.ml_models.enabled_models,
            'supported_formats': self.get_supported_formats(),
            'extractors': list(self.extractors.keys()),
            'privacy_settings': {
                'redaction_enabled': settings.privacy.enable_redaction,
                'gdpr_compliance': settings.privacy.gdpr_compliance,
                'law25_compliance': settings.privacy.law25_compliance
            }
        }
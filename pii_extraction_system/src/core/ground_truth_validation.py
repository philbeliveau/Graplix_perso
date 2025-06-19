"""
Ground Truth Validation Tools for Phase 0 Dataset Creation

This module provides validation tools for ground truth labeling including:
- Confidence scoring algorithms
- Inter-annotator agreement calculation
- Label quality assessment
- Validation workflow management
- Human-in-the-loop validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

class GroundTruthValidator:
    """Main class for ground truth validation operations"""
    
    def __init__(self):
        self.validation_history = []
        self.confidence_threshold = 0.8
        self.agreement_threshold = 0.75
        
    def validate_document_labels(
        self, 
        document: Dict[str, Any],
        validation_method: str = "confidence_based"
    ) -> Dict[str, Any]:
        """
        Validate labels for a single document
        
        Args:
            document: Document with labels to validate
            validation_method: Method to use for validation
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_result = {
            'document_id': document.get('id'),
            'validation_method': validation_method,
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'entity_scores': [],
            'issues_detected': [],
            'recommendations': [],
            'needs_human_review': False
        }
        
        if not document.get('gpt4o_labels'):
            validation_result['issues_detected'].append({
                'type': 'missing_labels',
                'severity': 'high',
                'description': 'Document has no labels to validate'
            })
            return validation_result
        
        labels = document['gpt4o_labels']
        entities = labels.get('entities', [])
        
        if validation_method == "confidence_based":
            validation_result = self._validate_by_confidence(entities, validation_result)
        elif validation_method == "consistency_based":
            validation_result = self._validate_by_consistency(entities, validation_result)
        elif validation_method == "completeness_based":
            validation_result = self._validate_by_completeness(document, validation_result)
        
        # Determine if human review is needed
        validation_result['needs_human_review'] = (
            validation_result['overall_score'] < self.confidence_threshold or
            len([issue for issue in validation_result['issues_detected'] 
                 if issue['severity'] in ['high', 'critical']]) > 0
        )
        
        # Add to validation history
        self.validation_history.append(validation_result)
        
        return validation_result
    
    def _validate_by_confidence(
        self, 
        entities: List[Dict], 
        validation_result: Dict
    ) -> Dict:
        """Validate entities based on confidence scores"""
        if not entities:
            validation_result['overall_score'] = 0.0
            validation_result['issues_detected'].append({
                'type': 'no_entities',
                'severity': 'medium',
                'description': 'No entities found in document'
            })
            return validation_result
        
        confidence_scores = []
        low_confidence_entities = []
        
        for i, entity in enumerate(entities):
            confidence = entity.get('confidence', 0.0)
            confidence_scores.append(confidence)
            
            entity_score = {
                'entity_index': i,
                'entity_type': entity.get('type'),
                'entity_text': entity.get('text'),
                'confidence': confidence,
                'validation_score': confidence,
                'issues': []
            }
            
            if confidence < 0.5:
                entity_score['issues'].append('very_low_confidence')
                low_confidence_entities.append(entity)
            elif confidence < self.confidence_threshold:
                entity_score['issues'].append('low_confidence')
                low_confidence_entities.append(entity)
            
            validation_result['entity_scores'].append(entity_score)
        
        # Calculate overall confidence score
        validation_result['overall_score'] = np.mean(confidence_scores)
        
        # Add issues for low confidence entities
        if low_confidence_entities:
            validation_result['issues_detected'].append({
                'type': 'low_confidence_entities',
                'severity': 'medium' if len(low_confidence_entities) < len(entities) * 0.3 else 'high',
                'description': f'{len(low_confidence_entities)} entities have low confidence scores',
                'affected_entities': len(low_confidence_entities)
            })
            
            validation_result['recommendations'].append(
                'Review and manually validate entities with confidence < 80%'
            )
        
        return validation_result
    
    def _validate_by_consistency(
        self, 
        entities: List[Dict], 
        validation_result: Dict
    ) -> Dict:
        """Validate entities based on consistency patterns"""
        entity_type_patterns = {}
        inconsistencies = []
        
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            entity_text = entity.get('text', '')
            
            if entity_type not in entity_type_patterns:
                entity_type_patterns[entity_type] = {
                    'texts': [],
                    'patterns': set(),
                    'confidence_scores': []
                }
            
            entity_type_patterns[entity_type]['texts'].append(entity_text)
            entity_type_patterns[entity_type]['confidence_scores'].append(
                entity.get('confidence', 0.0)
            )
            
            # Analyze patterns for specific entity types
            if entity_type == 'EMAIL':
                if '@' not in entity_text or '.' not in entity_text.split('@')[-1]:
                    inconsistencies.append({
                        'type': 'invalid_email_format',
                        'entity': entity,
                        'description': f'Email "{entity_text}" has invalid format'
                    })
            
            elif entity_type == 'PHONE':
                # Basic phone number validation
                cleaned_phone = ''.join(filter(str.isdigit, entity_text))
                if len(cleaned_phone) < 10 or len(cleaned_phone) > 15:
                    inconsistencies.append({
                        'type': 'invalid_phone_format',
                        'entity': entity,
                        'description': f'Phone "{entity_text}" has unusual format'
                    })
        
        # Calculate consistency score
        consistency_scores = []
        for entity_type, patterns in entity_type_patterns.items():
            # Calculate variance in confidence scores for same type
            if len(patterns['confidence_scores']) > 1:
                variance = np.var(patterns['confidence_scores'])
                consistency_score = max(0, 1.0 - variance)
                consistency_scores.append(consistency_score)
        
        validation_result['overall_score'] = np.mean(consistency_scores) if consistency_scores else 0.8
        
        # Add inconsistency issues
        for inconsistency in inconsistencies:
            validation_result['issues_detected'].append({
                'type': inconsistency['type'],
                'severity': 'medium',
                'description': inconsistency['description']
            })
        
        if inconsistencies:
            validation_result['recommendations'].append(
                'Review entities with format inconsistencies'
            )
        
        return validation_result
    
    def _validate_by_completeness(
        self, 
        document: Dict, 
        validation_result: Dict
    ) -> Dict:
        """Validate based on expected completeness"""
        labels = document.get('gpt4o_labels', {})
        entities = labels.get('entities', [])
        transcribed_text = labels.get('transcribed_text', '')
        
        # Expected patterns based on document type
        doc_type = document.get('metadata', {}).get('document_type', 'Unknown')
        expected_entities = self._get_expected_entities_for_doc_type(doc_type)
        
        found_types = set(entity.get('type') for entity in entities)
        missing_types = expected_entities - found_types
        
        if missing_types:
            validation_result['issues_detected'].append({
                'type': 'missing_expected_entities',
                'severity': 'medium',
                'description': f'Expected entity types not found: {", ".join(missing_types)}',
                'missing_types': list(missing_types)
            })
            
            validation_result['recommendations'].append(
                f'Review document for missing {", ".join(missing_types)} entities'
            )
        
        # Check for text coverage
        if transcribed_text:
            text_length = len(transcribed_text)
            entity_coverage = sum(len(entity.get('text', '')) for entity in entities)
            coverage_ratio = entity_coverage / text_length if text_length > 0 else 0
            
            if coverage_ratio < 0.1:  # Less than 10% of text is PII
                validation_result['issues_detected'].append({
                    'type': 'low_pii_coverage',
                    'severity': 'low',
                    'description': f'Only {coverage_ratio:.1%} of text identified as PII'
                })
        
        # Calculate completeness score
        completeness_score = 1.0 - (len(missing_types) / max(1, len(expected_entities)))
        validation_result['overall_score'] = completeness_score
        
        return validation_result
    
    def _get_expected_entities_for_doc_type(self, doc_type: str) -> set:
        """Get expected entity types for document type"""
        expected_entities_map = {
            'HR': {'PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'ID_NUMBER'},
            'Finance': {'PERSON', 'ORGANIZATION', 'ADDRESS', 'PHONE', 'EMAIL'},
            'Healthcare': {'PERSON', 'DATE', 'ID_NUMBER', 'ADDRESS'},
            'Legal': {'PERSON', 'ORGANIZATION', 'DATE', 'ADDRESS'},
            'PDF Document': {'PERSON', 'EMAIL'},
            'Word Document': {'PERSON', 'EMAIL'},
            'Image': {'PERSON', 'EMAIL', 'PHONE'},
            'Unknown': {'PERSON'}
        }
        
        return expected_entities_map.get(doc_type, {'PERSON'})
    
    def calculate_inter_annotator_agreement(
        self, 
        annotations_list: List[List[Dict]]
    ) -> Dict[str, float]:
        """
        Calculate inter-annotator agreement between multiple annotation sets
        
        Args:
            annotations_list: List of annotation sets from different annotators
            
        Returns:
            Agreement scores using various metrics
        """
        if len(annotations_list) < 2:
            return {'error': 'At least 2 annotation sets required'}
        
        # Convert annotations to comparable format
        standardized_annotations = []
        for annotations in annotations_list:
            standardized = self._standardize_annotations(annotations)
            standardized_annotations.append(standardized)
        
        # Calculate different agreement metrics
        agreement_scores = {}
        
        # Entity-level agreement (presence/absence)
        entity_agreement = self._calculate_entity_agreement(standardized_annotations)
        agreement_scores['entity_agreement'] = entity_agreement
        
        # Type-level agreement (for matching entities)
        type_agreement = self._calculate_type_agreement(standardized_annotations)
        agreement_scores['type_agreement'] = type_agreement
        
        # Text-level agreement (exact text match)
        text_agreement = self._calculate_text_agreement(standardized_annotations)
        agreement_scores['text_agreement'] = text_agreement
        
        # Overall Kappa score
        if len(standardized_annotations) == 2:
            kappa_score = self._calculate_kappa_score(
                standardized_annotations[0], 
                standardized_annotations[1]
            )
            agreement_scores['kappa_score'] = kappa_score
        
        return agreement_scores
    
    def _standardize_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Standardize annotation format for comparison"""
        standardized = []
        for annotation in annotations:
            standardized.append({
                'type': annotation.get('type', '').upper(),
                'text': annotation.get('text', '').strip().lower(),
                'confidence': annotation.get('confidence', 0.0),
                'start': annotation.get('start', 0),
                'end': annotation.get('end', 0)
            })
        return standardized
    
    def _calculate_entity_agreement(
        self, 
        annotations_list: List[List[Dict]]
    ) -> float:
        """Calculate agreement on entity presence"""
        if not annotations_list:
            return 0.0
        
        # Get all unique entity positions across annotators
        all_positions = set()
        for annotations in annotations_list:
            for ann in annotations:
                position = (ann.get('start', 0), ann.get('end', 0))
                all_positions.add(position)
        
        if not all_positions:
            return 1.0  # Perfect agreement if no entities found by anyone
        
        agreements = 0
        total_comparisons = 0
        
        for pos in all_positions:
            annotator_decisions = []
            for annotations in annotations_list:
                has_entity = any(
                    ann.get('start', 0) == pos[0] and ann.get('end', 0) == pos[1]
                    for ann in annotations
                )
                annotator_decisions.append(has_entity)
            
            # Count agreements
            for i in range(len(annotator_decisions)):
                for j in range(i + 1, len(annotator_decisions)):
                    total_comparisons += 1
                    if annotator_decisions[i] == annotator_decisions[j]:
                        agreements += 1
        
        return agreements / total_comparisons if total_comparisons > 0 else 1.0
    
    def _calculate_type_agreement(
        self, 
        annotations_list: List[List[Dict]]
    ) -> float:
        """Calculate agreement on entity types for matching positions"""
        matching_entities = self._find_matching_entities(annotations_list)
        
        if not matching_entities:
            return 1.0
        
        agreements = 0
        total_comparisons = 0
        
        for entity_group in matching_entities:
            types = [entity['type'] for entity in entity_group if entity is not None]
            
            # Count pairwise agreements
            for i in range(len(types)):
                for j in range(i + 1, len(types)):
                    total_comparisons += 1
                    if types[i] == types[j]:
                        agreements += 1
        
        return agreements / total_comparisons if total_comparisons > 0 else 1.0
    
    def _calculate_text_agreement(
        self, 
        annotations_list: List[List[Dict]]
    ) -> float:
        """Calculate agreement on entity text for matching positions"""
        matching_entities = self._find_matching_entities(annotations_list)
        
        if not matching_entities:
            return 1.0
        
        agreements = 0
        total_comparisons = 0
        
        for entity_group in matching_entities:
            texts = [entity['text'] for entity in entity_group if entity is not None]
            
            # Count pairwise agreements
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    total_comparisons += 1
                    if texts[i] == texts[j]:
                        agreements += 1
        
        return agreements / total_comparisons if total_comparisons > 0 else 1.0
    
    def _find_matching_entities(
        self, 
        annotations_list: List[List[Dict]]
    ) -> List[List[Optional[Dict]]]:
        """Find entities that match across annotators (by position)"""
        if not annotations_list:
            return []
        
        # Get all unique positions
        all_positions = set()
        for annotations in annotations_list:
            for ann in annotations:
                position = (ann.get('start', 0), ann.get('end', 0))
                all_positions.add(position)
        
        matching_groups = []
        for pos in all_positions:
            group = []
            for annotations in annotations_list:
                matching_entity = None
                for ann in annotations:
                    if (ann.get('start', 0), ann.get('end', 0)) == pos:
                        matching_entity = ann
                        break
                group.append(matching_entity)
            
            # Only include groups where at least 2 annotators found an entity
            if sum(1 for entity in group if entity is not None) >= 2:
                matching_groups.append(group)
        
        return matching_groups
    
    def _calculate_kappa_score(
        self, 
        annotations1: List[Dict], 
        annotations2: List[Dict]
    ) -> float:
        """Calculate Cohen's Kappa score between two annotation sets"""
        try:
            # Create binary vectors for entity presence at each position
            all_positions = set()
            
            for ann in annotations1 + annotations2:
                position = (ann.get('start', 0), ann.get('end', 0))
                all_positions.add(position)
            
            if not all_positions:
                return 1.0
            
            annotator1_labels = []
            annotator2_labels = []
            
            for pos in sorted(all_positions):
                # Check if annotator 1 found entity at this position
                has_entity1 = any(
                    (ann.get('start', 0), ann.get('end', 0)) == pos
                    for ann in annotations1
                )
                
                # Check if annotator 2 found entity at this position
                has_entity2 = any(
                    (ann.get('start', 0), ann.get('end', 0)) == pos
                    for ann in annotations2
                )
                
                annotator1_labels.append(int(has_entity1))
                annotator2_labels.append(int(has_entity2))
            
            return cohen_kappa_score(annotator1_labels, annotator2_labels)
            
        except Exception as e:
            logger.warning(f"Could not calculate Kappa score: {e}")
            return 0.0
    
    def generate_validation_report(
        self, 
        documents: List[Dict],
        include_detailed_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for dataset
        
        Args:
            documents: List of documents with labels
            include_detailed_scores: Whether to include per-document scores
            
        Returns:
            Comprehensive validation report
        """
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_documents': len(documents),
                'validation_version': '1.0'
            },
            'summary_statistics': {},
            'quality_metrics': {},
            'issues_summary': {},
            'recommendations': [],
            'detailed_scores': [] if include_detailed_scores else None
        }
        
        if not documents:
            return report
        
        # Validate all documents
        validation_results = []
        for document in documents:
            result = self.validate_document_labels(document)
            validation_results.append(result)
            
            if include_detailed_scores:
                report['detailed_scores'].append({
                    'document_id': document.get('id'),
                    'document_name': document.get('name'),
                    'validation_score': result['overall_score'],
                    'issues_count': len(result['issues_detected']),
                    'needs_review': result['needs_human_review']
                })
        
        # Calculate summary statistics
        scores = [result['overall_score'] for result in validation_results]
        report['summary_statistics'] = {
            'mean_validation_score': np.mean(scores),
            'median_validation_score': np.median(scores),
            'std_validation_score': np.std(scores),
            'min_validation_score': np.min(scores),
            'max_validation_score': np.max(scores),
            'documents_needing_review': sum(
                1 for result in validation_results 
                if result['needs_human_review']
            ),
            'high_quality_documents': sum(
                1 for score in scores if score >= self.confidence_threshold
            )
        }
        
        # Aggregate quality metrics
        all_issues = []
        for result in validation_results:
            all_issues.extend(result['issues_detected'])
        
        issue_types = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            severity = issue.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report['issues_summary'] = {
            'total_issues': len(all_issues),
            'issue_types': issue_types,
            'severity_distribution': severity_counts
        }
        
        # Generate recommendations
        recommendations = set()
        for result in validation_results:
            recommendations.update(result['recommendations'])
        
        report['recommendations'] = list(recommendations)
        
        # Add quality assessment
        overall_quality = report['summary_statistics']['mean_validation_score']
        quality_assessment = self._assess_overall_quality(overall_quality, severity_counts)
        report['quality_metrics'] = quality_assessment
        
        return report
    
    def _assess_overall_quality(
        self, 
        mean_score: float, 
        severity_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Assess overall dataset quality"""
        assessment = {
            'overall_grade': 'F',
            'quality_score': mean_score,
            'quality_level': 'Poor',
            'major_concerns': [],
            'strengths': [],
            'improvement_areas': []
        }
        
        # Grade assignment
        if mean_score >= 0.9:
            assessment['overall_grade'] = 'A'
            assessment['quality_level'] = 'Excellent'
        elif mean_score >= 0.8:
            assessment['overall_grade'] = 'B'
            assessment['quality_level'] = 'Good'
        elif mean_score >= 0.7:
            assessment['overall_grade'] = 'C'
            assessment['quality_level'] = 'Fair'
        elif mean_score >= 0.6:
            assessment['overall_grade'] = 'D'
            assessment['quality_level'] = 'Poor'
        else:
            assessment['overall_grade'] = 'F'
            assessment['quality_level'] = 'Very Poor'
        
        # Identify concerns and strengths
        if severity_counts.get('critical', 0) > 0:
            assessment['major_concerns'].append('Critical validation issues detected')
        
        if severity_counts.get('high', 0) > severity_counts.get('low', 0):
            assessment['major_concerns'].append('High number of high-severity issues')
        
        if mean_score >= 0.8:
            assessment['strengths'].append('High overall validation scores')
        
        if severity_counts.get('low', 0) > severity_counts.get('high', 0):
            assessment['strengths'].append('Most issues are low severity')
        
        # Improvement recommendations
        if mean_score < 0.8:
            assessment['improvement_areas'].append('Increase overall validation scores')
        
        if severity_counts.get('high', 0) + severity_counts.get('critical', 0) > 0:
            assessment['improvement_areas'].append('Address high-severity validation issues')
        
        return assessment
    
    def create_human_review_queue(
        self, 
        documents: List[Dict],
        prioritization_method: str = "validation_score"
    ) -> List[Dict]:
        """
        Create prioritized queue for human review
        
        Args:
            documents: Documents to potentially review
            prioritization_method: Method for prioritizing review queue
            
        Returns:
            Ordered list of documents for human review
        """
        review_queue = []
        
        for document in documents:
            validation_result = self.validate_document_labels(document)
            
            if validation_result['needs_human_review']:
                review_item = {
                    'document_id': document.get('id'),
                    'document_name': document.get('name'),
                    'validation_score': validation_result['overall_score'],
                    'issues_count': len(validation_result['issues_detected']),
                    'high_severity_issues': len([
                        issue for issue in validation_result['issues_detected']
                        if issue.get('severity') in ['high', 'critical']
                    ]),
                    'estimated_review_time': self._estimate_review_time(document, validation_result),
                    'priority_score': self._calculate_priority_score(document, validation_result),
                    'validation_details': validation_result
                }
                review_queue.append(review_item)
        
        # Sort by priority method
        if prioritization_method == "validation_score":
            review_queue.sort(key=lambda x: x['validation_score'])
        elif prioritization_method == "issues_count":
            review_queue.sort(key=lambda x: x['issues_count'], reverse=True)
        elif prioritization_method == "priority_score":
            review_queue.sort(key=lambda x: x['priority_score'], reverse=True)
        elif prioritization_method == "estimated_time":
            review_queue.sort(key=lambda x: x['estimated_review_time'])
        
        return review_queue
    
    def _estimate_review_time(
        self, 
        document: Dict, 
        validation_result: Dict
    ) -> int:
        """Estimate time needed for human review (in minutes)"""
        base_time = 5  # Base 5 minutes per document
        
        # Add time based on number of entities
        entity_count = len(
            document.get('gpt4o_labels', {}).get('entities', [])
        )
        entity_time = entity_count * 0.5  # 30 seconds per entity
        
        # Add time based on issues
        issues_time = len(validation_result['issues_detected']) * 2  # 2 minutes per issue
        
        # Add time based on document complexity  
        complexity = document.get('metadata', {}).get('complexity_score', 5)
        complexity_time = max(0, (complexity - 3) * 2)  # Additional time for complex docs
        
        total_time = base_time + entity_time + issues_time + complexity_time
        return min(30, max(5, int(total_time)))  # Cap between 5-30 minutes
    
    def _calculate_priority_score(
        self, 
        document: Dict, 
        validation_result: Dict
    ) -> float:
        """Calculate priority score for review queue"""
        score = 0.0
        
        # Lower validation score = higher priority
        score += (1.0 - validation_result['overall_score']) * 10
        
        # More high-severity issues = higher priority
        high_severity_count = len([
            issue for issue in validation_result['issues_detected']
            if issue.get('severity') in ['high', 'critical']
        ])
        score += high_severity_count * 5
        
        # Document priority
        doc_priority = document.get('priority', 'Medium')
        priority_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        score += priority_weights.get(doc_priority, 2) * 2
        
        # Sensitive data gets higher priority
        if document.get('metadata', {}).get('contains_sensitive', False):
            score += 3
        
        return score

# Global validator instance
ground_truth_validator = GroundTruthValidator()
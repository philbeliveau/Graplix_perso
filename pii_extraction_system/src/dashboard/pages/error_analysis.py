"""
Error Analysis Page - Analyze and correct PII extraction errors

This page provides comprehensive error analysis interfaces for identifying
false positives, false negatives, and systematic model failures.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth

def show_page():
    """Main error analysis page"""
    st.markdown('<div class="section-header">🐛 Error Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("Analyze extraction errors and provide feedback for model improvement.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Error Detection",
        "📊 Error Patterns", 
        "✏️ Manual Correction",
        "📈 Feedback Analytics"
    ])
    
    with tab1:
        show_error_detection()
    
    with tab2:
        show_error_patterns()
    
    with tab3:
        show_manual_correction()
    
    with tab4:
        show_feedback_analytics()

def show_error_detection():
    """Error detection and categorization interface"""
    st.markdown("### Error Detection")
    
    # Document selection
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    if not uploaded_docs:
        st.info("Upload and process documents to analyze errors.")
        return
    
    doc_options = {doc_id: info['name'] for doc_id, info in uploaded_docs.items() 
                  if session_state.get_processing_results(doc_id)}
    
    if not doc_options:
        st.info("No processed documents available for error analysis.")
        return
    
    selected_doc = st.selectbox(
        "Select document for error analysis:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    if not selected_doc:
        return
    
    results = session_state.get_processing_results(selected_doc)
    if not results:
        st.error("No processing results found.")
        return
    
    # Show document with error annotation interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_document_with_error_marking(selected_doc, results)
    
    with col2:
        show_error_summary(selected_doc)

def show_document_with_error_marking(doc_id: str, results: Dict[str, Any]):
    """Show document with interface for marking errors"""
    st.markdown("#### Document with Error Marking")
    
    text_content = results.get('text_content', '')
    pii_entities = results.get('pii_entities', [])
    
    if not text_content:
        st.warning("No text content available.")
        return
    
    # Confidence threshold for filtering
    confidence_threshold = st.slider(
        "Show entities above confidence:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower threshold to see more potential errors"
    )
    
    # Filter entities by confidence
    filtered_entities = [
        entity for entity in pii_entities 
        if entity.get('confidence', 0) >= confidence_threshold
    ]
    
    # Show highlighted text with click-to-correct functionality
    if filtered_entities:
        st.markdown("**Click on highlighted entities to mark as errors:**")
        
        # Create interactive highlighted text
        highlighted_html = create_interactive_highlights(text_content, filtered_entities, doc_id)
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # Entity selection for error marking
        st.markdown("---")
        st.markdown("**Mark Entities as Errors:**")
        
        for i, entity in enumerate(filtered_entities):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.text(f"{entity.get('type', 'UNKNOWN')}: {entity.get('text', '')}")
            
            with col2:
                st.text(f"Conf: {entity.get('confidence', 0):.2f}")
            
            with col3:
                if st.button(f"Mark Error", key=f"error_{doc_id}_{i}"):
                    mark_entity_as_error(doc_id, entity, "false_positive")
                    st.rerun()
    else:
        st.text_area("Document Text", text_content, height=400, disabled=True)
    
    # Interface for marking missed entities (false negatives)
    st.markdown("---")
    st.markdown("#### Mark Missed PII (False Negatives)")
    
    with st.form(f"missed_pii_{doc_id}"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            missed_text = st.text_input("PII text that was missed:")
        
        with col2:
            pii_categories = ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 
                             'CREDIT_CARD', 'DATE', 'ORGANIZATION', 'OTHER']
            missed_category = st.selectbox("PII Category:", pii_categories)
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            submitted = st.form_submit_button("Add Missed PII")
        
        if submitted and missed_text:
            add_missed_pii(doc_id, missed_text, missed_category)
            st.success("Missed PII added for analysis!")
            st.rerun()

def create_interactive_highlights(text: str, entities: List[Dict], doc_id: str) -> str:
    """Create interactive highlighted text"""
    # Sort entities by start position (reverse order for replacement)
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
    
    highlighted_text = text
    for i, entity in enumerate(sorted_entities):
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        entity_type = entity.get('type', 'UNKNOWN')
        confidence = entity.get('confidence', 0.0)
        
        # Color based on confidence
        if confidence > 0.8:
            color = '#99ff99'  # Green for high confidence
        elif confidence > 0.6:
            color = '#ffff99'  # Yellow for medium confidence
        else:
            color = '#ff9999'  # Red for low confidence (likely errors)
        
        # Create highlighted span with click handler
        highlighted_span = f"""
        <span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; 
                     border: 1px solid #ccc; cursor: pointer;" 
              title="{entity_type} (Confidence: {confidence:.2f}) - Click to mark as error"
              onclick="markError('{doc_id}', {i})">
            {text[start:end]}
        </span>
        """
        
        highlighted_text = (highlighted_text[:start] + highlighted_span + 
                          highlighted_text[end:])
    
    return f'<div style="line-height: 1.6; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{highlighted_text}</div>'

def mark_entity_as_error(doc_id: str, entity: Dict[str, Any], error_type: str):
    """Mark an entity as an error"""
    error_annotation = {
        'entity': entity,
        'error_type': error_type,
        'corrected_by': st.session_state.get('username', 'unknown'),
        'notes': ''
    }
    
    session_state.add_error_annotation(doc_id, error_annotation)

def add_missed_pii(doc_id: str, text: str, category: str):
    """Add a missed PII entity (false negative)"""
    missed_entity = {
        'text': text,
        'type': category,
        'confidence': 1.0,  # Ground truth
        'start': -1,  # Not found in original
        'end': -1
    }
    
    error_annotation = {
        'entity': missed_entity,
        'error_type': 'false_negative',
        'corrected_by': st.session_state.get('username', 'unknown'),
        'notes': 'User-identified missed entity'
    }
    
    session_state.add_error_annotation(doc_id, error_annotation)

def show_error_summary(doc_id: str):
    """Show error summary for selected document"""
    st.markdown("#### Error Summary")
    
    error_annotations = session_state.get_error_annotations(doc_id)
    
    if not error_annotations:
        st.info("No errors marked yet.")
        return
    
    # Count errors by type
    error_counts = {}
    for annotation in error_annotations:
        error_type = annotation.get('error_type', 'unknown')
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    # Display error metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("False Positives", error_counts.get('false_positive', 0))
    
    with col2:
        st.metric("False Negatives", error_counts.get('false_negative', 0))
    
    # Recent error annotations
    st.markdown("**Recent Annotations:**")
    for i, annotation in enumerate(error_annotations[-5:]):  # Show last 5
        entity = annotation.get('entity', {})
        error_type = annotation.get('error_type', 'unknown')
        
        st.write(f"• {error_type}: {entity.get('type', 'UNKNOWN')} - {entity.get('text', 'N/A')}")

def show_error_patterns():
    """Show error pattern analysis"""
    st.markdown("### Error Pattern Analysis")
    
    # Collect all error annotations across documents
    all_errors = collect_all_error_annotations()
    
    if not all_errors:
        st.info("No error annotations available for pattern analysis.")
        return
    
    # Analyze patterns
    col1, col2 = st.columns(2)
    
    with col1:
        show_error_by_category(all_errors)
    
    with col2:
        show_error_by_confidence(all_errors)
    
    # Detailed pattern analysis
    st.markdown("---")
    show_detailed_pattern_analysis(all_errors)

def collect_all_error_annotations() -> List[Dict[str, Any]]:
    """Collect all error annotations from all documents"""
    all_errors = []
    
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    for doc_id in uploaded_docs.keys():
        errors = session_state.get_error_annotations(doc_id)
        for error in errors:
            error['document_id'] = doc_id
            all_errors.append(error)
    
    return all_errors

def show_error_by_category(errors: List[Dict[str, Any]]):
    """Show error distribution by PII category"""
    st.markdown("#### Errors by PII Category")
    
    category_errors = {}
    for error in errors:
        entity = error.get('entity', {})
        category = entity.get('type', 'UNKNOWN')
        error_type = error.get('error_type', 'unknown')
        
        if category not in category_errors:
            category_errors[category] = {'false_positive': 0, 'false_negative': 0}
        
        category_errors[category][error_type] = category_errors[category].get(error_type, 0) + 1
    
    if category_errors:
        # Create chart
        chart_data = []
        for category, counts in category_errors.items():
            chart_data.append({
                'Category': category,
                'False Positives': counts.get('false_positive', 0),
                'False Negatives': counts.get('false_negative', 0)
            })
        
        df = pd.DataFrame(chart_data)
        st.bar_chart(df.set_index('Category'))

def show_error_by_confidence(errors: List[Dict[str, Any]]):
    """Show error distribution by confidence level"""
    st.markdown("#### Errors by Confidence Level")
    
    confidence_bins = {'0.0-0.3': 0, '0.3-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
    
    for error in errors:
        if error.get('error_type') == 'false_positive':  # Only for false positives
            entity = error.get('entity', {})
            confidence = entity.get('confidence', 0)
            
            if confidence < 0.3:
                confidence_bins['0.0-0.3'] += 1
            elif confidence < 0.6:
                confidence_bins['0.3-0.6'] += 1
            elif confidence < 0.8:
                confidence_bins['0.6-0.8'] += 1
            else:
                confidence_bins['0.8-1.0'] += 1
    
    # Display as bar chart
    df = pd.DataFrame(list(confidence_bins.items()), columns=['Confidence Range', 'Error Count'])
    st.bar_chart(df.set_index('Confidence Range'))

def show_detailed_pattern_analysis(errors: List[Dict[str, Any]]):
    """Show detailed pattern analysis"""
    st.markdown("#### Detailed Pattern Analysis")
    
    # Most common error patterns
    error_patterns = {}
    
    for error in errors:
        entity = error.get('entity', {})
        pattern_key = f"{entity.get('type', 'UNKNOWN')}_{error.get('error_type', 'unknown')}"
        
        if pattern_key not in error_patterns:
            error_patterns[pattern_key] = {
                'count': 0,
                'examples': [],
                'avg_confidence': 0,
                'confidence_scores': []
            }
        
        error_patterns[pattern_key]['count'] += 1
        error_patterns[pattern_key]['examples'].append(entity.get('text', 'N/A'))
        
        if entity.get('confidence'):
            error_patterns[pattern_key]['confidence_scores'].append(entity.get('confidence'))
    
    # Calculate average confidence for each pattern
    for pattern in error_patterns.values():
        if pattern['confidence_scores']:
            pattern['avg_confidence'] = sum(pattern['confidence_scores']) / len(pattern['confidence_scores'])
    
    # Sort by frequency
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Display top patterns
    st.markdown("**Most Common Error Patterns:**")
    
    for pattern_key, pattern_data in sorted_patterns[:10]:  # Top 10
        with st.expander(f"{pattern_key} ({pattern_data['count']} occurrences)"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Examples:**")
                for example in pattern_data['examples'][:5]:  # Show up to 5 examples
                    st.write(f"• {example}")
            
            with col2:
                if pattern_data['avg_confidence'] > 0:
                    st.metric("Average Confidence", f"{pattern_data['avg_confidence']:.2%}")
                
                # Recommendations
                st.write("**Recommendations:**")
                if 'false_positive' in pattern_key and pattern_data['avg_confidence'] < 0.5:
                    st.write("• Consider raising confidence threshold")
                elif 'false_negative' in pattern_key:
                    st.write("• Review extraction patterns")
                    st.write("• Consider additional training data")

def show_manual_correction():
    """Manual correction interface"""
    st.markdown("### Manual Correction Interface")
    
    if not auth.has_permission('write'):
        st.warning("Manual correction requires write permissions.")
        return
    
    # Document selection
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    processed_docs = {doc_id: info for doc_id, info in uploaded_docs.items() 
                     if session_state.get_processing_results(doc_id)}
    
    if not processed_docs:
        st.info("No processed documents available for correction.")
        return
    
    doc_options = {doc_id: info['name'] for doc_id, info in processed_docs.items()}
    selected_doc = st.selectbox(
        "Select document to correct:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    if not selected_doc:
        return
    
    results = session_state.get_processing_results(selected_doc)
    if not results:
        st.error("No processing results found.")
        return
    
    # Correction interface
    show_correction_interface(selected_doc, results)

def show_correction_interface(doc_id: str, results: Dict[str, Any]):
    """Show interface for manual corrections"""
    st.markdown("#### Correction Interface")
    
    pii_entities = results.get('pii_entities', [])
    error_annotations = session_state.get_error_annotations(doc_id)
    
    # Show entities that need correction
    corrections_made = []
    
    st.markdown("**Review and Correct PII Entities:**")
    
    for i, entity in enumerate(pii_entities):
        # Check if this entity has been marked as an error
        is_error = any(
            ann.get('entity', {}).get('text') == entity.get('text') and 
            ann.get('error_type') == 'false_positive'
            for ann in error_annotations
        )
        
        if is_error or entity.get('confidence', 1.0) < 0.6:  # Show low confidence or marked errors
            with st.expander(f"{'❌ ' if is_error else '⚠️ '}{entity.get('type', 'UNKNOWN')}: {entity.get('text', '')}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Correction options
                    correction_action = st.radio(
                        "Action:",
                        ["Keep as-is", "Remove (False Positive)", "Change Category", "Change Text"],
                        key=f"action_{doc_id}_{i}"
                    )
                
                with col2:
                    if correction_action == "Change Category":
                        new_category = st.selectbox(
                            "New Category:",
                            ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 
                             'CREDIT_CARD', 'DATE', 'ORGANIZATION', 'OTHER'],
                            key=f"category_{doc_id}_{i}"
                        )
                    elif correction_action == "Change Text":
                        new_text = st.text_input(
                            "Corrected Text:",
                            value=entity.get('text', ''),
                            key=f"text_{doc_id}_{i}"
                        )
                
                with col3:
                    confidence_override = st.slider(
                        "Confidence:",
                        min_value=0.0,
                        max_value=1.0,
                        value=entity.get('confidence', 0.5),
                        key=f"conf_{doc_id}_{i}"
                    )
                
                # Apply correction
                if st.button(f"Apply Correction", key=f"apply_{doc_id}_{i}"):
                    correction = {
                        'original_entity': entity,
                        'action': correction_action,
                        'new_category': new_category if correction_action == "Change Category" else None,
                        'new_text': new_text if correction_action == "Change Text" else None,
                        'new_confidence': confidence_override,
                        'corrected_by': st.session_state.get('username', 'unknown')
                    }
                    
                    apply_correction(doc_id, i, correction)
                    st.success("Correction applied!")
                    st.rerun()

def apply_correction(doc_id: str, entity_index: int, correction: Dict[str, Any]):
    """Apply manual correction to entity"""
    results = session_state.get_processing_results(doc_id)
    if not results:
        return
    
    pii_entities = results.get('pii_entities', [])
    if entity_index >= len(pii_entities):
        return
    
    action = correction.get('action')
    
    if action == "Remove (False Positive)":
        # Remove the entity
        pii_entities.pop(entity_index)
    elif action == "Change Category":
        # Update category
        pii_entities[entity_index]['type'] = correction.get('new_category')
        pii_entities[entity_index]['confidence'] = correction.get('new_confidence')
    elif action == "Change Text":
        # Update text
        pii_entities[entity_index]['text'] = correction.get('new_text')
        pii_entities[entity_index]['confidence'] = correction.get('new_confidence')
    
    # Store updated results
    results['pii_entities'] = pii_entities
    session_state.store_processing_results(doc_id, results)
    
    # Log the correction
    correction_log = {
        'entity_index': entity_index,
        'correction': correction,
        'timestamp': st.session_state.get('current_time', 'now')
    }
    
    if 'correction_log' not in st.session_state:
        st.session_state.correction_log = {}
    if doc_id not in st.session_state.correction_log:
        st.session_state.correction_log[doc_id] = []
    
    st.session_state.correction_log[doc_id].append(correction_log)

def show_feedback_analytics():
    """Show feedback analytics and improvement suggestions"""
    st.markdown("### Feedback Analytics")
    
    all_errors = collect_all_error_annotations()
    all_corrections = collect_all_corrections()
    
    if not all_errors and not all_corrections:
        st.info("No feedback data available yet.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Errors Identified", len(all_errors))
    
    with col2:
        fp_count = sum(1 for e in all_errors if e.get('error_type') == 'false_positive')
        st.metric("False Positives", fp_count)
    
    with col3:
        fn_count = sum(1 for e in all_errors if e.get('error_type') == 'false_negative')
        st.metric("False Negatives", fn_count)
    
    with col4:
        st.metric("Manual Corrections", len(all_corrections))
    
    # Feedback trends
    st.markdown("---")
    show_feedback_trends(all_errors, all_corrections)
    
    # Improvement recommendations
    st.markdown("---")
    show_improvement_recommendations(all_errors, all_corrections)

def collect_all_corrections() -> List[Dict[str, Any]]:
    """Collect all manual corrections"""
    all_corrections = []
    
    correction_log = st.session_state.get('correction_log', {})
    for doc_id, corrections in correction_log.items():
        for correction in corrections:
            correction['document_id'] = doc_id
            all_corrections.append(correction)
    
    return all_corrections

def show_feedback_trends(errors: List[Dict], corrections: List[Dict]):
    """Show feedback trends over time"""
    st.markdown("#### Feedback Trends")
    
    # Mock trend data (in production, would track actual timestamps)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    error_counts = np.random.poisson(2, 30)  # Random error counts
    correction_counts = np.random.poisson(1, 30)  # Random correction counts
    
    trend_data = pd.DataFrame({
        'Date': dates,
        'Errors Identified': error_counts,
        'Corrections Made': correction_counts
    })
    
    st.line_chart(trend_data.set_index('Date'))

def show_improvement_recommendations(errors: List[Dict], corrections: List[Dict]):
    """Show improvement recommendations based on feedback"""
    st.markdown("#### Improvement Recommendations")
    
    recommendations = []
    
    # Analyze error patterns for recommendations
    if errors:
        fp_count = sum(1 for e in errors if e.get('error_type') == 'false_positive')
        fn_count = sum(1 for e in errors if e.get('error_type') == 'false_negative')
        
        if fp_count > fn_count * 2:
            recommendations.append({
                'priority': 'High',
                'category': 'Precision',
                'description': 'High false positive rate detected',
                'action': 'Consider increasing confidence thresholds or improving precision of models'
            })
        
        if fn_count > fp_count * 2:
            recommendations.append({
                'priority': 'High', 
                'category': 'Recall',
                'description': 'High false negative rate detected',
                'action': 'Consider lowering confidence thresholds or improving recall of models'
            })
    
    # Category-specific recommendations
    category_errors = {}
    for error in errors:
        entity = error.get('entity', {})
        category = entity.get('type', 'UNKNOWN')
        if category not in category_errors:
            category_errors[category] = 0
        category_errors[category] += 1
    
    # Find categories with most errors
    if category_errors:
        max_errors = max(category_errors.values())
        problematic_categories = [cat for cat, count in category_errors.items() 
                                if count >= max_errors * 0.7]
        
        for category in problematic_categories:
            recommendations.append({
                'priority': 'Medium',
                'category': f'{category} Detection',
                'description': f'High error rate in {category} category',
                'action': f'Review and improve {category} detection patterns or training data'
            })
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': 'red',
                'Medium': 'orange', 
                'Low': 'green'
            }.get(rec['priority'], 'gray')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
                <strong>{rec['priority']} Priority - {rec['category']}</strong><br>
                {rec['description']}<br>
                <em>Recommended Action:</em> {rec['action']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations available yet. Continue providing feedback to generate insights.")
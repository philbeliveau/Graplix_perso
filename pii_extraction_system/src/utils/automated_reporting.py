"""
Automated Report Generation System

This module provides comprehensive automated report generation capabilities
for multimodal LLM performance analysis, including executive summaries,
technical analyses, and actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import io
import base64
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of automated reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_ANALYSIS = "technical_analysis"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_COMPARISON = "performance_comparison"
    VARIANCE_ANALYSIS = "variance_analysis"
    RECOMMENDATIONS = "recommendations"
    FULL_REPORT = "full_report"

class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    PDF = "pdf"

@dataclass
class ReportMetadata:
    """Report metadata and configuration"""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    data_sources: List[str]
    total_records: int
    filters_applied: Dict[str, Any]
    configuration: Dict[str, Any]

@dataclass
class KeyInsight:
    """Key insight or finding"""
    category: str
    severity: str  # "info", "warning", "critical", "success"
    title: str
    description: str
    impact: str
    recommendation: str
    supporting_data: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics summary"""
    total_tests: int
    average_accuracy: float
    total_cost: float
    average_latency: float
    success_rate: float
    models_tested: int
    documents_processed: int
    period_days: int

@dataclass
class ReportContent:
    """Complete report content structure"""
    metadata: ReportMetadata
    executive_summary: str
    key_insights: List[KeyInsight]
    performance_metrics: PerformanceMetrics
    detailed_analysis: Dict[str, Any]
    visualizations: Dict[str, str]  # Chart name -> base64 encoded image
    recommendations: List[Dict[str, Any]]
    appendices: Dict[str, Any]

class AutomatedReportGenerator:
    """Main automated report generation system"""
    
    def __init__(self):
        self.templates = self._load_report_templates()
        self.visualization_engine = VisualizationEngine()
        self.insight_engine = InsightEngine()
        self.recommendation_engine = RecommendationEngine()
    
    def generate_report(self, 
                       data: pd.DataFrame,
                       report_type: ReportType,
                       time_period: Optional[Tuple[datetime, datetime]] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       configuration: Optional[Dict[str, Any]] = None) -> ReportContent:
        """
        Generate a comprehensive automated report
        
        Args:
            data: Performance data DataFrame
            report_type: Type of report to generate
            time_period: Optional time period for analysis
            filters: Optional filters to apply to data
            configuration: Optional report configuration
        
        Returns:
            Complete report content
        """
        
        # Set defaults
        if time_period is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            time_period = (start_date, end_date)
        
        if filters is None:
            filters = {}
        
        if configuration is None:
            configuration = self._get_default_configuration(report_type)
        
        # Apply filters to data
        filtered_data = self._apply_filters(data, filters, time_period)
        
        # Generate metadata
        metadata = self._generate_metadata(
            report_type, time_period, filtered_data, filters, configuration
        )
        
        # Generate performance metrics
        performance_metrics = self._calculate_performance_metrics(filtered_data, time_period)
        
        # Generate key insights
        key_insights = self.insight_engine.generate_insights(filtered_data, report_type)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            filtered_data, report_type, configuration
        )
        
        # Generate visualizations
        visualizations = self.visualization_engine.generate_report_visualizations(
            filtered_data, report_type, configuration
        )
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            filtered_data, key_insights, report_type
        )
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            performance_metrics, key_insights, recommendations
        )
        
        # Generate appendices
        appendices = self._generate_appendices(filtered_data, configuration)
        
        return ReportContent(
            metadata=metadata,
            executive_summary=executive_summary,
            key_insights=key_insights,
            performance_metrics=performance_metrics,
            detailed_analysis=detailed_analysis,
            visualizations=visualizations,
            recommendations=recommendations,
            appendices=appendices
        )
    
    def export_report(self, 
                     report_content: ReportContent, 
                     format: ReportFormat,
                     output_path: Optional[str] = None) -> str:
        """
        Export report to specified format
        
        Args:
            report_content: Generated report content
            format: Output format
            output_path: Optional output file path
        
        Returns:
            Exported content as string or file path
        """
        
        if format == ReportFormat.JSON:
            return self._export_json(report_content, output_path)
        elif format == ReportFormat.HTML:
            return self._export_html(report_content, output_path)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report_content, output_path)
        elif format == ReportFormat.PDF:
            return self._export_pdf(report_content, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_report_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates for different report types"""
        templates = {}
        
        # Executive Summary Template
        templates['executive_summary'] = Template("""
# Executive Summary

## Overview
This report covers {{ period_days }} days of multimodal LLM performance analysis 
from {{ start_date }} to {{ end_date }}.

## Key Performance Indicators
- **Total Tests Conducted**: {{ total_tests:,}}
- **Average Accuracy**: {{ average_accuracy }}}
- **Total Processing Cost**: ${{ total_cost }}
- **Average Latency**: {{ average_latency }}s
- **Overall Success Rate**: {{ success_rate }}

## Critical Insights
{% for insight in critical_insights %}
- **{{ insight.title }}**: {{ insight.description }}
{% endfor %}

## Priority Recommendations
{% for rec in priority_recommendations %}
{{ loop.index }}. **{{ rec.title }}**: {{ rec.description }}
{% endfor %}
        """)
        
        # Technical Analysis Template
        templates['technical_analysis'] = Template("""
# Technical Performance Analysis

## Model Performance Breakdown
{% for model, metrics in model_performance.items() %}
### {{ model }}
- Accuracy: {{ metrics.accuracy }}
- Cost Efficiency: {{ metrics.cost_efficiency }}
- Latency: {{ metrics.latency }}s
- Adaptability: {{ metrics.adaptability }}
{% endfor %}

## Statistical Analysis
{{ statistical_summary }}

## Variance Analysis
{{ variance_analysis }}
        """)
        
        return templates
    
    def _get_default_configuration(self, report_type: ReportType) -> Dict[str, Any]:
        """Get default configuration for report type"""
        configs = {
            ReportType.EXECUTIVE_SUMMARY: {
                'include_charts': True,
                'detail_level': 'high_level',
                'focus_areas': ['performance', 'cost', 'recommendations']
            },
            ReportType.TECHNICAL_ANALYSIS: {
                'include_charts': True,
                'detail_level': 'detailed',
                'include_statistical_tests': True,
                'focus_areas': ['accuracy', 'latency', 'variance']
            },
            ReportType.COST_OPTIMIZATION: {
                'include_charts': True,
                'detail_level': 'detailed',
                'focus_areas': ['cost', 'efficiency', 'optimization']
            },
            ReportType.FULL_REPORT: {
                'include_charts': True,
                'detail_level': 'comprehensive',
                'include_all_sections': True,
                'focus_areas': ['all']
            }
        }
        
        return configs.get(report_type, {
            'include_charts': True,
            'detail_level': 'medium',
            'focus_areas': ['performance', 'cost']
        })
    
    def _apply_filters(self, 
                      data: pd.DataFrame, 
                      filters: Dict[str, Any], 
                      time_period: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Apply filters to the data"""
        filtered_data = data.copy()
        
        # Time period filter
        if 'timestamp' in filtered_data.columns:
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
            start_date, end_date = time_period
            filtered_data = filtered_data[
                (filtered_data['timestamp'] >= start_date) & 
                (filtered_data['timestamp'] <= end_date)
            ]
        
        # Model filter
        if 'models' in filters and 'model' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['model'].isin(filters['models'])]
        
        # Difficulty filter
        if 'difficulty_levels' in filters and 'difficulty_level' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['difficulty_level'].isin(filters['difficulty_levels'])]
        
        # Success only filter
        if filters.get('success_only', False) and 'success' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['success'] == True]
        
        # Cost range filter
        if 'cost_range' in filters and 'cost' in filtered_data.columns:
            min_cost, max_cost = filters['cost_range']
            filtered_data = filtered_data[
                (filtered_data['cost'] >= min_cost) & 
                (filtered_data['cost'] <= max_cost)
            ]
        
        return filtered_data
    
    def _generate_metadata(self, 
                          report_type: ReportType,
                          time_period: Tuple[datetime, datetime],
                          data: pd.DataFrame,
                          filters: Dict[str, Any],
                          configuration: Dict[str, Any]) -> ReportMetadata:
        """Generate report metadata"""
        
        # Determine data sources
        data_sources = ['multimodal_test_results']
        if 'run_history' in str(data.columns):
            data_sources.append('run_history')
        
        return ReportMetadata(
            report_id=f"report_{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=report_type,
            generated_at=datetime.now(),
            time_period=time_period,
            data_sources=data_sources,
            total_records=len(data),
            filters_applied=filters,
            configuration=configuration
        )
    
    def _calculate_performance_metrics(self, 
                                     data: pd.DataFrame, 
                                     time_period: Tuple[datetime, datetime]) -> PerformanceMetrics:
        """Calculate overall performance metrics"""
        
        if len(data) == 0:
            return PerformanceMetrics(
                total_tests=0, average_accuracy=0, total_cost=0, average_latency=0,
                success_rate=0, models_tested=0, documents_processed=0,
                period_days=(time_period[1] - time_period[0]).days
            )
        
        # Calculate metrics
        total_tests = len(data)
        
        average_accuracy = data['accuracy_score'].mean() if 'accuracy_score' in data.columns else 0
        
        total_cost = data['cost'].sum() if 'cost' in data.columns else 0
        
        average_latency = data['latency'].mean() if 'latency' in data.columns else 0
        
        success_rate = data['success'].mean() if 'success' in data.columns else 1
        
        models_tested = data['model'].nunique() if 'model' in data.columns else 0
        
        documents_processed = data['document_name'].nunique() if 'document_name' in data.columns else total_tests
        
        period_days = (time_period[1] - time_period[0]).days
        
        return PerformanceMetrics(
            total_tests=total_tests,
            average_accuracy=average_accuracy,
            total_cost=total_cost,
            average_latency=average_latency,
            success_rate=success_rate,
            models_tested=models_tested,
            documents_processed=documents_processed,
            period_days=period_days
        )
    
    def _generate_detailed_analysis(self, 
                                  data: pd.DataFrame,
                                  report_type: ReportType,
                                  configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis section"""
        
        analysis = {}
        
        # Model-level analysis
        if 'model' in data.columns:
            model_analysis = data.groupby('model').agg({
                'accuracy_score': ['mean', 'std', 'count'] if 'accuracy_score' in data.columns else 'count',
                'cost': ['mean', 'sum'] if 'cost' in data.columns else 'count',
                'latency': ['mean', 'std'] if 'latency' in data.columns else 'count'
            }).round(4)
            
            analysis['model_performance'] = model_analysis.to_dict()
        
        # Difficulty analysis
        if 'difficulty_level' in data.columns:
            difficulty_analysis = data.groupby('difficulty_level').agg({
                'accuracy_score': 'mean' if 'accuracy_score' in data.columns else 'count',
                'success': 'mean' if 'success' in data.columns else 'count'
            }).round(4)
            
            analysis['difficulty_analysis'] = difficulty_analysis.to_dict()
        
        # Temporal analysis
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['date'] = data['timestamp'].dt.date
            
            daily_analysis = data.groupby('date').agg({
                'accuracy_score': 'mean' if 'accuracy_score' in data.columns else 'count',
                'cost': 'sum' if 'cost' in data.columns else 'count'
            }).round(4)
            
            analysis['temporal_trends'] = daily_analysis.to_dict()
        
        # Cost efficiency analysis
        if 'cost' in data.columns and 'accuracy_score' in data.columns:
            data['cost_efficiency'] = data['accuracy_score'] / data['cost']
            efficiency_analysis = data.groupby('model')['cost_efficiency'].agg(['mean', 'std']).round(4)
            analysis['cost_efficiency'] = efficiency_analysis.to_dict()
        
        # Statistical summary
        if configuration.get('include_statistical_tests', False):
            analysis['statistical_summary'] = self._generate_statistical_summary(data)
        
        return analysis
    
    def _generate_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary and tests"""
        summary = {}
        
        # Descriptive statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        summary['descriptive_stats'] = data[numeric_columns].describe().to_dict()
        
        # Correlation analysis
        if len(numeric_columns) > 1:
            correlation_matrix = data[numeric_columns].corr()
            summary['correlations'] = correlation_matrix.to_dict()
        
        # Model comparison tests
        if 'model' in data.columns and 'accuracy_score' in data.columns:
            from scipy import stats
            
            models = data['model'].unique()
            if len(models) > 1:
                # ANOVA test for accuracy differences
                model_groups = [data[data['model'] == model]['accuracy_score'].dropna() 
                              for model in models]
                
                model_groups = [group for group in model_groups if len(group) > 0]
                
                if len(model_groups) > 1:
                    try:
                        f_stat, p_value = stats.f_oneway(*model_groups)
                        summary['anova_test'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        summary['anova_test'] = {'error': 'Could not perform ANOVA test'}
        
        return summary
    
    def _generate_executive_summary(self, 
                                  performance_metrics: PerformanceMetrics,
                                  key_insights: List[KeyInsight],
                                  recommendations: List[Dict[str, Any]]) -> str:
        """Generate executive summary text"""
        
        # Get critical insights and priority recommendations
        critical_insights = [insight for insight in key_insights 
                           if insight.severity in ['critical', 'warning']]
        
        priority_recommendations = recommendations[:3]  # Top 3 recommendations
        
        # Use template
        template = self.templates['executive_summary']
        
        summary = template.render(
            period_days=performance_metrics.period_days,
            start_date=datetime.now() - timedelta(days=performance_metrics.period_days),
            end_date=datetime.now(),
            total_tests=performance_metrics.total_tests,
            average_accuracy=f"{performance_metrics.average_accuracy:.1%}",
            total_cost=f"{performance_metrics.total_cost:.4f}",
            average_latency=f"{performance_metrics.average_latency:.1f}",
            success_rate=f"{performance_metrics.success_rate:.1%}",
            critical_insights=critical_insights,
            priority_recommendations=priority_recommendations
        )
        
        return summary
    
    def _generate_appendices(self, 
                           data: pd.DataFrame, 
                           configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report appendices"""
        
        appendices = {}
        
        # Data quality assessment
        appendices['data_quality'] = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict(),
            'unique_values': {col: data[col].nunique() for col in data.columns}
        }
        
        # Methodology
        appendices['methodology'] = {
            'analysis_period': configuration.get('time_period', 'Last 7 days'),
            'metrics_calculated': [
                'Accuracy (mean confidence scores)',
                'Cost (total and per-document)',
                'Latency (processing time)',
                'Adaptability (performance consistency)',
                'Success Rate (% successful extractions)'
            ],
            'statistical_methods': [
                'Descriptive statistics',
                'ANOVA for model comparison',
                'Correlation analysis',
                'Outlier detection (IQR method)'
            ]
        }
        
        # Glossary
        appendices['glossary'] = {
            'Accuracy': 'Average confidence score of extracted entities',
            'Adaptability': 'Consistency of performance across different document types',
            'Cost Efficiency': 'Accuracy achieved per unit cost',
            'Latency': 'Time taken to process a single document',
            'Pareto Efficient': 'Model that cannot be improved in one dimension without degrading another',
            'Composite Score': 'Weighted combination of accuracy, cost, and adaptability metrics'
        }
        
        return appendices
    
    def _export_json(self, report_content: ReportContent, output_path: Optional[str]) -> str:
        """Export report as JSON"""
        
        # Convert dataclasses to dictionaries
        report_dict = asdict(report_content)
        
        # Handle datetime serialization
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json_content = json.dumps(report_dict, indent=2, default=datetime_handler)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_content)
            return output_path
        
        return json_content
    
    def _export_html(self, report_content: ReportContent, output_path: Optional[str]) -> str:
        """Export report as HTML"""
        
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ metadata.report_type.value.title() }} Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .insight { margin: 15px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }
        .insight.warning { border-left-color: #ffc107; }
        .insight.critical { border-left-color: #dc3545; }
        .recommendation { margin: 10px 0; padding: 15px; background: #d4edda; border-radius: 5px; }
        .chart-container { margin: 20px 0; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ metadata.report_type.value.replace('_', ' ').title() }} Report</h1>
        <p>Generated on {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p>Analysis Period: {{ metadata.time_period[0].strftime('%Y-%m-%d') }} to {{ metadata.time_period[1].strftime('%Y-%m-%d') }}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ performance_metrics.total_tests }}</div>
            <div>Total Tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.1f%%"|format(performance_metrics.average_accuracy * 100) }}</div>
            <div>Average Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${{ "%.4f"|format(performance_metrics.total_cost) }}</div>
            <div>Total Cost</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.1f"|format(performance_metrics.average_latency) }}s</div>
            <div>Average Latency</div>
        </div>
    </div>
    
    <h2>Executive Summary</h2>
    <div>{{ executive_summary|replace('\\n', '<br>')|safe }}</div>
    
    <h2>Key Insights</h2>
    {% for insight in key_insights %}
    <div class="insight {{ insight.severity }}">
        <h4>{{ insight.title }}</h4>
        <p>{{ insight.description }}</p>
        <small><strong>Impact:</strong> {{ insight.impact }}</small>
    </div>
    {% endfor %}
    
    <h2>Recommendations</h2>
    {% for rec in recommendations %}
    <div class="recommendation">
        <h4>{{ rec.title }}</h4>
        <p>{{ rec.description }}</p>
        {% if rec.get('expected_impact') %}
        <small><strong>Expected Impact:</strong> {{ rec.expected_impact }}</small>
        {% endif %}
    </div>
    {% endfor %}
    
    {% if visualizations %}
    <h2>Visualizations</h2>
    {% for chart_name, chart_data in visualizations.items() %}
    <div class="chart-container">
        <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
        <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}" style="max-width: 100%;">
    </div>
    {% endfor %}
    {% endif %}
    
</body>
</html>
        """)
        
        html_content = html_template.render(
            metadata=report_content.metadata,
            performance_metrics=report_content.performance_metrics,
            executive_summary=report_content.executive_summary,
            key_insights=report_content.key_insights,
            recommendations=report_content.recommendations,
            visualizations=report_content.visualizations
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            return output_path
        
        return html_content
    
    def _export_markdown(self, report_content: ReportContent, output_path: Optional[str]) -> str:
        """Export report as Markdown"""
        
        md_content = f"""# {report_content.metadata.report_type.value.replace('_', ' ').title()} Report

Generated on {report_content.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {report_content.metadata.time_period[0].strftime('%Y-%m-%d')} to {report_content.metadata.time_period[1].strftime('%Y-%m-%d')}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Tests | {report_content.performance_metrics.total_tests:,} |
| Average Accuracy | {report_content.performance_metrics.average_accuracy:.1%} |
| Total Cost | ${report_content.performance_metrics.total_cost:.4f} |
| Average Latency | {report_content.performance_metrics.average_latency:.1f}s |
| Success Rate | {report_content.performance_metrics.success_rate:.1%} |

## Executive Summary

{report_content.executive_summary}

## Key Insights

"""
        
        for insight in report_content.key_insights:
            severity_emoji = {
                'info': 'ℹ️',
                'warning': '⚠️', 
                'critical': '🚨',
                'success': '✅'
            }.get(insight.severity, 'ℹ️')
            
            md_content += f"""
### {severity_emoji} {insight.title}

**Description:** {insight.description}

**Impact:** {insight.impact}

**Recommendation:** {insight.recommendation}

---
"""
        
        md_content += "\n## Recommendations\n"
        
        for i, rec in enumerate(report_content.recommendations, 1):
            md_content += f"""
### {i}. {rec.get('title', 'Recommendation')}

{rec.get('description', '')}

**Priority:** {rec.get('priority', 'Medium')}
**Expected Impact:** {rec.get('expected_impact', 'Not specified')}

"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(md_content)
            return output_path
        
        return md_content
    
    def _export_pdf(self, report_content: ReportContent, output_path: Optional[str]) -> str:
        """Export report as PDF (placeholder - would need additional dependencies)"""
        # This would require libraries like reportlab or weasyprint
        # For now, return HTML export with PDF indication
        html_content = self._export_html(report_content, None)
        
        if output_path:
            with open(output_path.replace('.pdf', '.html'), 'w') as f:
                f.write(html_content)
            return output_path.replace('.pdf', '.html')
        
        return html_content

class VisualizationEngine:
    """Engine for generating report visualizations"""
    
    def generate_report_visualizations(self, 
                                     data: pd.DataFrame,
                                     report_type: ReportType,
                                     configuration: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for the report"""
        
        if not configuration.get('include_charts', True):
            return {}
        
        visualizations = {}
        
        try:
            # Performance overview chart
            if 'model' in data.columns and 'accuracy_score' in data.columns:
                fig = self._create_performance_overview(data)
                visualizations['performance_overview'] = self._fig_to_base64(fig)
            
            # Cost analysis chart
            if 'cost' in data.columns:
                fig = self._create_cost_analysis(data)
                visualizations['cost_analysis'] = self._fig_to_base64(fig)
            
            # Temporal trends
            if 'timestamp' in data.columns:
                fig = self._create_temporal_trends(data)
                visualizations['temporal_trends'] = self._fig_to_base64(fig)
            
            # Model comparison
            if report_type in [ReportType.PERFORMANCE_COMPARISON, ReportType.FULL_REPORT]:
                fig = self._create_model_comparison(data)
                visualizations['model_comparison'] = self._fig_to_base64(fig)
        
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_performance_overview(self, data: pd.DataFrame) -> go.Figure:
        """Create performance overview chart"""
        
        if 'model' not in data.columns:
            return go.Figure()
        
        model_performance = data.groupby('model').agg({
            'accuracy_score': 'mean',
            'cost': 'mean',
            'latency': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=model_performance['model'],
            y=model_performance['accuracy_score'],
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            name='Cost',
            x=model_performance['model'],
            y=model_performance['cost'],
            yaxis='y2',
            mode='markers+lines',
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Model Performance Overview',
            xaxis=dict(title='Model'),
            yaxis=dict(title='Accuracy', side='left'),
            yaxis2=dict(title='Cost ($)', side='right', overlaying='y'),
            legend=dict(x=0.7, y=1),
            height=400
        )
        
        return fig
    
    def _create_cost_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create cost analysis chart"""
        
        if 'cost' not in data.columns:
            return go.Figure()
        
        fig = px.histogram(
            data, 
            x='cost', 
            title='Cost Distribution',
            labels={'cost': 'Cost per Document ($)', 'count': 'Frequency'}
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def _create_temporal_trends(self, data: pd.DataFrame) -> go.Figure:
        """Create temporal trends chart"""
        
        if 'timestamp' not in data.columns:
            return go.Figure()
        
        data_copy = data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        daily_metrics = data_copy.groupby('date').agg({
            'accuracy_score': 'mean',
            'cost': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Average Accuracy', 'Daily Total Cost'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['accuracy_score'],
                      mode='lines+markers', name='Accuracy'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_metrics['date'], y=daily_metrics['cost'],
                      mode='lines+markers', name='Cost'),
            row=2, col=1
        )
        
        fig.update_layout(title='Performance Trends Over Time', height=500)
        
        return fig
    
    def _create_model_comparison(self, data: pd.DataFrame) -> go.Figure:
        """Create detailed model comparison chart"""
        
        if 'model' not in data.columns:
            return go.Figure()
        
        model_metrics = data.groupby('model').agg({
            'accuracy_score': ['mean', 'std'],
            'cost': 'mean',
            'latency': 'mean'
        }).round(4)
        
        model_metrics.columns = ['Accuracy_Mean', 'Accuracy_Std', 'Cost_Mean', 'Latency_Mean']
        model_metrics = model_metrics.reset_index()
        
        fig = px.scatter(
            model_metrics,
            x='Cost_Mean',
            y='Accuracy_Mean',
            size='Latency_Mean',
            hover_name='model',
            title='Model Performance Matrix: Cost vs Accuracy (bubble size = latency)',
            labels={
                'Cost_Mean': 'Average Cost ($)',
                'Accuracy_Mean': 'Average Accuracy',
                'Latency_Mean': 'Average Latency (s)'
            }
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _fig_to_base64(self, fig: go.Figure) -> str:
        """Convert plotly figure to base64 string"""
        img_bytes = fig.to_image(format="png", width=800, height=500)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

class InsightEngine:
    """Engine for generating automated insights"""
    
    def generate_insights(self, 
                         data: pd.DataFrame, 
                         report_type: ReportType) -> List[KeyInsight]:
        """Generate key insights from the data"""
        
        insights = []
        
        # Performance insights
        insights.extend(self._generate_performance_insights(data))
        
        # Cost insights
        insights.extend(self._generate_cost_insights(data))
        
        # Model comparison insights
        insights.extend(self._generate_model_insights(data))
        
        # Quality insights
        insights.extend(self._generate_quality_insights(data))
        
        # Variance insights
        insights.extend(self._generate_variance_insights(data))
        
        return insights
    
    def _generate_performance_insights(self, data: pd.DataFrame) -> List[KeyInsight]:
        """Generate performance-related insights"""
        insights = []
        
        if 'accuracy_score' in data.columns and len(data) > 0:
            avg_accuracy = data['accuracy_score'].mean()
            
            if avg_accuracy > 0.9:
                insights.append(KeyInsight(
                    category="Performance",
                    severity="success",
                    title="Excellent Accuracy Performance",
                    description=f"Average accuracy of {avg_accuracy:.1%} exceeds industry benchmarks",
                    impact="High confidence in production deployment",
                    recommendation="Maintain current model selection and configuration"
                ))
            elif avg_accuracy < 0.7:
                insights.append(KeyInsight(
                    category="Performance",
                    severity="critical",
                    title="Low Accuracy Alert",
                    description=f"Average accuracy of {avg_accuracy:.1%} is below acceptable threshold",
                    impact="Risk of poor user experience and operational issues",
                    recommendation="Review model selection, increase training data, or implement ensemble methods"
                ))
        
        return insights
    
    def _generate_cost_insights(self, data: pd.DataFrame) -> List[KeyInsight]:
        """Generate cost-related insights"""
        insights = []
        
        if 'cost' in data.columns and len(data) > 0:
            total_cost = data['cost'].sum()
            avg_cost = data['cost'].mean()
            cost_variance = data['cost'].std() / avg_cost if avg_cost > 0 else 0
            
            if cost_variance > 0.5:
                insights.append(KeyInsight(
                    category="Cost",
                    severity="warning",
                    title="High Cost Variability",
                    description=f"Cost variance coefficient of {cost_variance:.1%} indicates inconsistent pricing",
                    impact="Unpredictable operational costs",
                    recommendation="Implement cost monitoring and consider standardizing on more predictable models"
                ))
            
            if avg_cost > 0.05:
                insights.append(KeyInsight(
                    category="Cost",
                    severity="warning",
                    title="High Per-Document Cost",
                    description=f"Average cost of ${avg_cost:.4f} per document may impact scalability",
                    impact="Limited scalability for high-volume processing",
                    recommendation="Evaluate cheaper models for simple documents or implement tiered processing"
                ))
        
        return insights
    
    def _generate_model_insights(self, data: pd.DataFrame) -> List[KeyInsight]:
        """Generate model comparison insights"""
        insights = []
        
        if 'model' in data.columns and len(data) > 0:
            model_count = data['model'].nunique()
            
            if model_count > 5:
                insights.append(KeyInsight(
                    category="Model Management",
                    severity="info",
                    title="Multiple Models in Use",
                    description=f"{model_count} different models being evaluated",
                    impact="Complexity in model management and decision-making",
                    recommendation="Consider consolidating to 2-3 top-performing models for production"
                ))
            
            # Find best and worst performers
            if 'accuracy_score' in data.columns:
                model_performance = data.groupby('model')['accuracy_score'].mean()
                best_model = model_performance.idxmax()
                worst_model = model_performance.idxmin()
                
                performance_gap = model_performance.max() - model_performance.min()
                
                if performance_gap > 0.2:
                    insights.append(KeyInsight(
                        category="Model Performance",
                        severity="warning",
                        title="Significant Performance Gap Between Models",
                        description=f"{performance_gap:.1%} difference between best ({best_model}) and worst ({worst_model}) models",
                        impact="Suboptimal performance if using lower-performing models",
                        recommendation=f"Prioritize {best_model} for production use and phase out {worst_model}"
                    ))
        
        return insights
    
    def _generate_quality_insights(self, data: pd.DataFrame) -> List[KeyInsight]:
        """Generate data quality insights"""
        insights = []
        
        # Success rate insights
        if 'success' in data.columns:
            success_rate = data['success'].mean()
            
            if success_rate < 0.95:
                insights.append(KeyInsight(
                    category="Quality",
                    severity="warning",
                    title="Processing Failures Detected",
                    description=f"Success rate of {success_rate:.1%} indicates processing issues",
                    impact="Potential data loss and operational disruptions",
                    recommendation="Investigate failure patterns and implement retry mechanisms"
                ))
        
        # Data completeness
        missing_data_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        if missing_data_ratio > 0.1:
            insights.append(KeyInsight(
                category="Data Quality",
                severity="warning",
                title="High Missing Data Rate",
                description=f"{missing_data_ratio:.1%} of data points are missing",
                impact="Reduced reliability of analysis and insights",
                recommendation="Review data collection processes and implement data validation"
            ))
        
        return insights
    
    def _generate_variance_insights(self, data: pd.DataFrame) -> List[KeyInsight]:
        """Generate variance and consistency insights"""
        insights = []
        
        if 'difficulty_level' in data.columns and 'accuracy_score' in data.columns:
            difficulty_performance = data.groupby('difficulty_level')['accuracy_score'].mean()
            
            if len(difficulty_performance) > 1:
                performance_variance = difficulty_performance.std()
                
                if performance_variance > 0.15:
                    insights.append(KeyInsight(
                        category="Adaptability",
                        severity="warning",
                        title="Inconsistent Performance Across Difficulty Levels",
                        description=f"Performance variance of {performance_variance:.1%} across difficulty levels",
                        impact="Unpredictable performance on diverse document types",
                        recommendation="Implement adaptive processing or difficulty-specific model selection"
                    ))
        
        return insights

class RecommendationEngine:
    """Engine for generating automated recommendations"""
    
    def generate_recommendations(self, 
                               data: pd.DataFrame,
                               insights: List[KeyInsight],
                               report_type: ReportType) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance optimization recommendations
        recommendations.extend(self._generate_performance_recommendations(data, insights))
        
        # Cost optimization recommendations
        recommendations.extend(self._generate_cost_recommendations(data, insights))
        
        # Model selection recommendations
        recommendations.extend(self._generate_model_recommendations(data, insights))
        
        # Operational recommendations
        recommendations.extend(self._generate_operational_recommendations(data, insights))
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2), reverse=True)
        
        return recommendations
    
    def _generate_performance_recommendations(self, 
                                            data: pd.DataFrame, 
                                            insights: List[KeyInsight]) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for accuracy issues
        accuracy_issues = [i for i in insights if i.category == "Performance" and i.severity in ["critical", "warning"]]
        
        if accuracy_issues:
            recommendations.append({
                'title': 'Implement Performance Monitoring',
                'description': 'Set up continuous monitoring of accuracy metrics with automated alerts for performance degradation',
                'priority': 'high',
                'category': 'Performance',
                'expected_impact': 'Early detection of issues, improved system reliability',
                'implementation_effort': 'medium',
                'timeline': '2-3 weeks'
            })
        
        # Model ensemble recommendation for low accuracy
        if 'accuracy_score' in data.columns:
            avg_accuracy = data['accuracy_score'].mean()
            if avg_accuracy < 0.8:
                recommendations.append({
                    'title': 'Consider Ensemble Method Implementation',
                    'description': 'Combine multiple models to improve overall accuracy and reduce variance',
                    'priority': 'high',
                    'category': 'Performance',
                    'expected_impact': '10-15% improvement in accuracy',
                    'implementation_effort': 'high',
                    'timeline': '4-6 weeks'
                })
        
        return recommendations
    
    def _generate_cost_recommendations(self, 
                                     data: pd.DataFrame, 
                                     insights: List[KeyInsight]) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if 'cost' in data.columns and 'model' in data.columns:
            # Find most cost-efficient model
            model_efficiency = data.groupby('model').apply(
                lambda x: x['accuracy_score'].mean() / x['cost'].mean() if x['cost'].mean() > 0 else 0
            )
            
            if len(model_efficiency) > 1:
                most_efficient = model_efficiency.idxmax()
                efficiency_ratio = model_efficiency.max() / model_efficiency.mean()
                
                if efficiency_ratio > 1.5:
                    recommendations.append({
                        'title': f'Standardize on {most_efficient} for Cost Efficiency',
                        'description': f'Model shows {efficiency_ratio:.1f}x better cost efficiency than average',
                        'priority': 'medium',
                        'category': 'Cost',
                        'expected_impact': f'Potential {(1-1/efficiency_ratio):.1%} cost reduction',
                        'implementation_effort': 'low',
                        'timeline': '1-2 weeks'
                    })
        
        # Tiered processing recommendation
        if 'difficulty_level' in data.columns:
            cost_by_difficulty = data.groupby('difficulty_level')['cost'].mean()
            
            if cost_by_difficulty.std() / cost_by_difficulty.mean() > 0.3:
                recommendations.append({
                    'title': 'Implement Tiered Processing Strategy',
                    'description': 'Use different models based on document difficulty to optimize costs',
                    'priority': 'medium',
                    'category': 'Cost',
                    'expected_impact': '20-30% cost reduction for simple documents',
                    'implementation_effort': 'medium',
                    'timeline': '3-4 weeks'
                })
        
        return recommendations
    
    def _generate_model_recommendations(self, 
                                      data: pd.DataFrame, 
                                      insights: List[KeyInsight]) -> List[Dict[str, Any]]:
        """Generate model selection recommendations"""
        recommendations = []
        
        if 'model' in data.columns and len(data) > 0:
            model_count = data['model'].nunique()
            
            if model_count > 5:
                recommendations.append({
                    'title': 'Consolidate Model Portfolio',
                    'description': f'Reduce from {model_count} models to 2-3 top performers for easier management',
                    'priority': 'medium',
                    'category': 'Model Management',
                    'expected_impact': 'Reduced complexity, easier monitoring and maintenance',
                    'implementation_effort': 'medium',
                    'timeline': '2-3 weeks'
                })
            
            # Recommend A/B testing for close performers
            if 'accuracy_score' in data.columns:
                model_performance = data.groupby('model')['accuracy_score'].mean().sort_values(ascending=False)
                
                if len(model_performance) > 1:
                    top_two_diff = model_performance.iloc[0] - model_performance.iloc[1]
                    
                    if top_two_diff < 0.05:  # Less than 5% difference
                        recommendations.append({
                            'title': 'Conduct A/B Testing for Top Models',
                            'description': f'Top models show similar performance ({top_two_diff:.1%} difference) - conduct controlled testing',
                            'priority': 'low',
                            'category': 'Model Selection',
                            'expected_impact': 'Data-driven model selection',
                            'implementation_effort': 'medium',
                            'timeline': '2-4 weeks'
                        })
        
        return recommendations
    
    def _generate_operational_recommendations(self, 
                                            data: pd.DataFrame, 
                                            insights: List[KeyInsight]) -> List[Dict[str, Any]]:
        """Generate operational recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_data_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        if missing_data_ratio > 0.05:
            recommendations.append({
                'title': 'Improve Data Collection Quality',
                'description': 'Implement data validation and quality checks to reduce missing data',
                'priority': 'medium',
                'category': 'Operations',
                'expected_impact': 'More reliable analysis and insights',
                'implementation_effort': 'medium',
                'timeline': '2-3 weeks'
            })
        
        # Monitoring recommendations
        recommendations.append({
            'title': 'Implement Comprehensive Monitoring Dashboard',
            'description': 'Create real-time dashboard for tracking performance, costs, and quality metrics',
            'priority': 'high',
            'category': 'Operations',
            'expected_impact': 'Proactive issue detection and faster response times',
            'implementation_effort': 'high',
            'timeline': '4-6 weeks'
        })
        
        return recommendations

# Global report generator instance
automated_report_generator = AutomatedReportGenerator()
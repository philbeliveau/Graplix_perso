"""
Cross-Domain Variance Analysis System

This module provides comprehensive variance analysis tools for evaluating model performance
consistency across different document domains, types, and complexity levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VarianceDimension(Enum):
    """Different dimensions for variance analysis"""
    DOCUMENT_TYPE = "document_type"
    DIFFICULTY_LEVEL = "difficulty_level"
    MODEL = "model"
    CONTENT_DOMAIN = "content_domain"
    FILE_SIZE = "file_size"
    IMAGE_QUALITY = "image_quality"
    TEMPORAL = "temporal"

class VarianceMetric(Enum):
    """Metrics for variance calculation"""
    ACCURACY = "accuracy"
    COST = "cost"
    LATENCY = "latency"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"

@dataclass
class VarianceResult:
    """Result of variance analysis"""
    dimension: VarianceDimension
    metric: VarianceMetric
    variance_score: float  # 0-1, higher = more variance
    stability_score: float  # 0-1, higher = more stable
    statistical_significance: float  # p-value
    effect_size: float  # Cohen's d or similar
    outliers: List[Dict[str, Any]]
    recommendations: List[str]
    detailed_stats: Dict[str, Any]

@dataclass
class CrossDomainAnalysis:
    """Comprehensive cross-domain analysis result"""
    overall_variance: float
    dimension_variances: Dict[VarianceDimension, float]
    interaction_effects: Dict[Tuple[VarianceDimension, VarianceDimension], float]
    stability_ranking: List[Tuple[str, float]]  # (model/condition, stability_score)
    performance_clusters: Dict[str, List[str]]
    variance_drivers: List[Dict[str, Any]]
    optimization_opportunities: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]

class VarianceAnalyzer:
    """Main variance analysis engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        
    def analyze_variance(self, 
                        data: pd.DataFrame, 
                        dimension: VarianceDimension, 
                        metric: VarianceMetric,
                        grouping_column: Optional[str] = None) -> VarianceResult:
        """
        Analyze variance for a specific dimension and metric
        
        Args:
            data: DataFrame containing performance data
            dimension: Dimension to analyze variance across
            metric: Performance metric to analyze
            grouping_column: Optional column to group by (if different from dimension)
        
        Returns:
            VarianceResult with comprehensive analysis
        """
        
        # Determine grouping column
        if grouping_column is None:
            grouping_column = dimension.value
        
        if grouping_column not in data.columns:
            raise ValueError(f"Column {grouping_column} not found in data")
        
        metric_column = self._get_metric_column(metric)
        
        if metric_column not in data.columns:
            raise ValueError(f"Metric column {metric_column} not found in data")
        
        # Clean data
        clean_data = data.dropna(subset=[grouping_column, metric_column])
        
        if len(clean_data) < 3:
            return self._create_empty_result(dimension, metric, "Insufficient data")
        
        # Group-wise analysis
        groups = clean_data.groupby(grouping_column)[metric_column]
        group_stats = groups.agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
        
        # Calculate variance metrics
        variance_score = self._calculate_variance_score(groups)
        stability_score = 1 - variance_score  # Inverse relationship
        
        # Statistical significance testing
        p_value = self._test_statistical_significance(groups)
        
        # Effect size calculation
        effect_size = self._calculate_effect_size(groups)
        
        # Outlier detection
        outliers = self._detect_outliers(clean_data, grouping_column, metric_column)
        
        # Generate recommendations
        recommendations = self._generate_variance_recommendations(
            dimension, metric, variance_score, group_stats, outliers
        )
        
        # Detailed statistics
        detailed_stats = {
            'group_statistics': group_stats.to_dict('records'),
            'overall_mean': clean_data[metric_column].mean(),
            'overall_std': clean_data[metric_column].std(),
            'coefficient_of_variation': clean_data[metric_column].std() / clean_data[metric_column].mean(),
            'range_ratio': (clean_data[metric_column].max() - clean_data[metric_column].min()) / clean_data[metric_column].mean(),
            'intergroup_variance': groups.std().std(),  # Variance of group standard deviations
            'sample_sizes': group_stats.set_index(grouping_column)['count'].to_dict()
        }
        
        return VarianceResult(
            dimension=dimension,
            metric=metric,
            variance_score=variance_score,
            stability_score=stability_score,
            statistical_significance=p_value,
            effect_size=effect_size,
            outliers=outliers,
            recommendations=recommendations,
            detailed_stats=detailed_stats
        )
    
    def analyze_cross_domain_variance(self, data: pd.DataFrame) -> CrossDomainAnalysis:
        """
        Comprehensive cross-domain variance analysis
        
        Args:
            data: DataFrame with performance data across multiple dimensions
        
        Returns:
            CrossDomainAnalysis with comprehensive insights
        """
        
        # Analyze variance across all available dimensions
        dimension_variances = {}
        
        available_dimensions = self._identify_available_dimensions(data)
        
        for dimension in available_dimensions:
            # Analyze variance for each metric
            metric_variances = []
            
            for metric in [VarianceMetric.ACCURACY, VarianceMetric.COST, VarianceMetric.LATENCY]:
                try:
                    result = self.analyze_variance(data, dimension, metric)
                    metric_variances.append(result.variance_score)
                except:
                    continue
            
            if metric_variances:
                dimension_variances[dimension] = np.mean(metric_variances)
        
        # Overall variance score
        overall_variance = np.mean(list(dimension_variances.values())) if dimension_variances else 0.5
        
        # Interaction effects analysis
        interaction_effects = self._analyze_interaction_effects(data, available_dimensions)
        
        # Stability ranking
        stability_ranking = self._calculate_stability_ranking(data)
        
        # Performance clustering
        performance_clusters = self._cluster_performance_profiles(data)
        
        # Identify variance drivers
        variance_drivers = self._identify_variance_drivers(data, dimension_variances)
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            data, dimension_variances, interaction_effects
        )
        
        # Risk assessment
        risk_assessment = self._assess_variance_risks(dimension_variances, overall_variance)
        
        return CrossDomainAnalysis(
            overall_variance=overall_variance,
            dimension_variances=dimension_variances,
            interaction_effects=interaction_effects,
            stability_ranking=stability_ranking,
            performance_clusters=performance_clusters,
            variance_drivers=variance_drivers,
            optimization_opportunities=optimization_opportunities,
            risk_assessment=risk_assessment
        )
    
    def _get_metric_column(self, metric: VarianceMetric) -> str:
        """Map variance metric to data column name"""
        mapping = {
            VarianceMetric.ACCURACY: 'accuracy_score',
            VarianceMetric.COST: 'cost',
            VarianceMetric.LATENCY: 'latency',
            VarianceMetric.CONSISTENCY: 'consistency_score',
            VarianceMetric.ROBUSTNESS: 'robustness_score'
        }
        return mapping.get(metric, 'accuracy_score')
    
    def _calculate_variance_score(self, groups) -> float:
        """Calculate normalized variance score (0-1)"""
        try:
            # Calculate coefficient of variation for each group
            group_cvs = []
            for name, group in groups:
                if len(group) > 1 and group.mean() != 0:
                    cv = group.std() / group.mean()
                    group_cvs.append(cv)
            
            if not group_cvs:
                return 0.5
            
            # Mean CV across groups
            mean_cv = np.mean(group_cvs)
            
            # Normalize to 0-1 scale (assuming CV > 2.0 is very high variance)
            variance_score = min(mean_cv / 2.0, 1.0)
            
            # Also consider between-group variance
            group_means = [group.mean() for name, group in groups]
            between_group_cv = np.std(group_means) / np.mean(group_means) if np.mean(group_means) != 0 else 0
            
            # Combine within-group and between-group variance
            combined_variance = 0.6 * variance_score + 0.4 * min(between_group_cv / 2.0, 1.0)
            
            return combined_variance
            
        except Exception as e:
            logger.warning(f"Error calculating variance score: {e}")
            return 0.5
    
    def _test_statistical_significance(self, groups) -> float:
        """Test statistical significance of differences between groups"""
        try:
            group_data = [group for name, group in groups if len(group) > 0]
            
            if len(group_data) < 2:
                return 1.0  # No significance with < 2 groups
            
            # Use ANOVA for multiple groups
            if len(group_data) > 2:
                statistic, p_value = stats.f_oneway(*group_data)
            else:
                # Use t-test for 2 groups
                statistic, p_value = stats.ttest_ind(group_data[0], group_data[1])
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Error in significance testing: {e}")
            return 1.0
    
    def _calculate_effect_size(self, groups) -> float:
        """Calculate effect size (Cohen's d or eta-squared)"""
        try:
            group_data = [group for name, group in groups if len(group) > 0]
            
            if len(group_data) < 2:
                return 0.0
            
            # For two groups, use Cohen's d
            if len(group_data) == 2:
                pooled_std = np.sqrt(((len(group_data[0]) - 1) * group_data[0].std()**2 + 
                                    (len(group_data[1]) - 1) * group_data[1].std()**2) / 
                                   (len(group_data[0]) + len(group_data[1]) - 2))
                
                if pooled_std == 0:
                    return 0.0
                
                cohens_d = abs(group_data[0].mean() - group_data[1].mean()) / pooled_std
                return cohens_d
            
            # For multiple groups, use eta-squared approximation
            else:
                all_data = np.concatenate(group_data)
                total_variance = np.var(all_data)
                
                if total_variance == 0:
                    return 0.0
                
                group_means = [group.mean() for group in group_data]
                between_group_variance = np.var(group_means)
                
                eta_squared = between_group_variance / total_variance
                return eta_squared
                
        except Exception as e:
            logger.warning(f"Error calculating effect size: {e}")
            return 0.0
    
    def _detect_outliers(self, data: pd.DataFrame, group_col: str, metric_col: str) -> List[Dict[str, Any]]:
        """Detect outliers in the data"""
        outliers = []
        
        try:
            # Use IQR method for outlier detection within each group
            for group_name in data[group_col].unique():
                group_data = data[data[group_col] == group_name][metric_col]
                
                if len(group_data) < 4:  # Need at least 4 points for IQR
                    continue
                
                Q1 = group_data.quantile(0.25)
                Q3 = group_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (group_data < lower_bound) | (group_data > upper_bound)
                group_outliers = data[(data[group_col] == group_name) & 
                                    (data[metric_col].isin(group_data[outlier_mask]))]
                
                for _, outlier in group_outliers.iterrows():
                    outliers.append({
                        'group': group_name,
                        'value': outlier[metric_col],
                        'expected_range': f"[{lower_bound:.3f}, {upper_bound:.3f}]",
                        'severity': 'high' if (outlier[metric_col] < lower_bound - IQR or 
                                             outlier[metric_col] > upper_bound + IQR) else 'moderate',
                        'data_point': outlier.to_dict()
                    })
            
        except Exception as e:
            logger.warning(f"Error detecting outliers: {e}")
        
        return outliers
    
    def _generate_variance_recommendations(self, 
                                         dimension: VarianceDimension, 
                                         metric: VarianceMetric,
                                         variance_score: float,
                                         group_stats: pd.DataFrame,
                                         outliers: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on variance analysis"""
        recommendations = []
        
        # General variance level recommendations
        if variance_score > 0.7:
            recommendations.append(f"🔴 **High Variance Alert**: {metric.value} shows significant variation across {dimension.value} - investigate root causes")
        elif variance_score > 0.4:
            recommendations.append(f"🟡 **Moderate Variance**: {metric.value} varies moderately across {dimension.value} - monitor closely")
        else:
            recommendations.append(f"🟢 **Low Variance**: {metric.value} is consistent across {dimension.value}")
        
        # Specific dimension recommendations
        if dimension == VarianceDimension.MODEL:
            if variance_score > 0.5:
                recommendations.append("🔧 **Model Selection**: Consider standardizing on more consistent models for production")
            
            # Find best and worst performing models
            best_model = group_stats.loc[group_stats['mean'].idxmax(), group_stats.columns[0]]
            worst_model = group_stats.loc[group_stats['mean'].idxmin(), group_stats.columns[0]]
            
            recommendations.append(f"🏆 **Best Performer**: {best_model} shows highest {metric.value}")
            
            if group_stats.loc[group_stats['mean'].idxmin(), 'mean'] < group_stats['mean'].mean() * 0.8:
                recommendations.append(f"⚠️ **Underperformer**: {worst_model} significantly below average - consider replacement")
        
        elif dimension == VarianceDimension.DIFFICULTY_LEVEL:
            if variance_score > 0.6:
                recommendations.append("📊 **Difficulty Adaptation**: Models struggle with certain difficulty levels - implement adaptive processing")
        
        elif dimension == VarianceDimension.DOCUMENT_TYPE:
            if variance_score > 0.5:
                recommendations.append("📄 **Document Specialization**: Consider specialized models for different document types")
        
        # Outlier-based recommendations
        if outliers:
            high_severity_outliers = [o for o in outliers if o['severity'] == 'high']
            if high_severity_outliers:
                recommendations.append(f"🎯 **Outlier Investigation**: {len(high_severity_outliers)} severe outliers detected - investigate data quality")
        
        # Sample size recommendations
        small_groups = group_stats[group_stats['count'] < 5]
        if len(small_groups) > 0:
            recommendations.append(f"📈 **Data Collection**: {len(small_groups)} groups have insufficient data (<5 samples) - increase testing")
        
        return recommendations
    
    def _identify_available_dimensions(self, data: pd.DataFrame) -> List[VarianceDimension]:
        """Identify which variance dimensions are available in the data"""
        available = []
        
        dimension_columns = {
            VarianceDimension.MODEL: 'model',
            VarianceDimension.DIFFICULTY_LEVEL: 'difficulty_level',
            VarianceDimension.DOCUMENT_TYPE: 'document_type',
            VarianceDimension.CONTENT_DOMAIN: 'content_domain',
            VarianceDimension.FILE_SIZE: 'file_size_category',
            VarianceDimension.IMAGE_QUALITY: 'image_quality_score'
        }
        
        for dimension, column in dimension_columns.items():
            if column in data.columns and data[column].notna().sum() > 0:
                available.append(dimension)
        
        # Temporal dimension if timestamp available
        if 'timestamp' in data.columns:
            available.append(VarianceDimension.TEMPORAL)
        
        return available
    
    def _analyze_interaction_effects(self, 
                                   data: pd.DataFrame, 
                                   dimensions: List[VarianceDimension]) -> Dict[Tuple[VarianceDimension, VarianceDimension], float]:
        """Analyze interaction effects between dimensions"""
        interactions = {}
        
        # Map dimensions to columns
        dimension_columns = {
            VarianceDimension.MODEL: 'model',
            VarianceDimension.DIFFICULTY_LEVEL: 'difficulty_level',
            VarianceDimension.DOCUMENT_TYPE: 'document_type'
        }
        
        # Only analyze dimensions that exist in data
        available_dims = [(d, dimension_columns[d]) for d in dimensions if d in dimension_columns and dimension_columns[d] in data.columns]
        
        for i, (dim1, col1) in enumerate(available_dims):
            for j, (dim2, col2) in enumerate(available_dims[i+1:], i+1):
                try:
                    # Create interaction groups
                    interaction_groups = data.groupby([col1, col2])['accuracy_score']
                    
                    if len(interaction_groups) < 2:
                        continue
                    
                    # Calculate interaction effect as variance in group means
                    group_means = interaction_groups.mean()
                    interaction_variance = group_means.std() / group_means.mean() if group_means.mean() != 0 else 0
                    
                    interactions[(dim1, dim2)] = min(interaction_variance, 1.0)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing interaction between {dim1} and {dim2}: {e}")
                    continue
        
        return interactions
    
    def _calculate_stability_ranking(self, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Calculate stability ranking for models or conditions"""
        stability_scores = []
        
        if 'model' not in data.columns:
            return stability_scores
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            if len(model_data) < 3:
                continue
            
            # Calculate stability as inverse of coefficient of variation
            metrics = ['accuracy_score', 'cost', 'latency']
            metric_stabilities = []
            
            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 1 and values.mean() != 0:
                        cv = values.std() / values.mean()
                        stability = 1 / (1 + cv)  # Higher stability for lower CV
                        metric_stabilities.append(stability)
            
            if metric_stabilities:
                overall_stability = np.mean(metric_stabilities)
                stability_scores.append((model, overall_stability))
        
        # Sort by stability (highest first)
        stability_scores.sort(key=lambda x: x[1], reverse=True)
        
        return stability_scores
    
    def _cluster_performance_profiles(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Cluster models/conditions by performance profiles"""
        clusters = {}
        
        if 'model' not in data.columns:
            return clusters
        
        try:
            # Create performance profiles for each model
            model_profiles = []
            model_names = []
            
            for model in data['model'].unique():
                model_data = data[data['model'] == model]
                
                if len(model_data) < 3:
                    continue
                
                # Create feature vector from performance metrics
                features = []
                
                # Accuracy metrics
                features.append(model_data['accuracy_score'].mean())
                features.append(model_data['accuracy_score'].std())
                
                # Cost metrics
                if 'cost' in model_data.columns:
                    features.append(model_data['cost'].mean())
                    features.append(model_data['cost'].std())
                
                # Latency metrics
                if 'latency' in model_data.columns:
                    features.append(model_data['latency'].mean())
                    features.append(model_data['latency'].std())
                
                if len(features) >= 4:  # Need minimum features for clustering
                    model_profiles.append(features)
                    model_names.append(model)
            
            if len(model_profiles) < 2:
                return clusters
            
            # Normalize features
            profiles_array = np.array(model_profiles)
            normalized_profiles = self.scaler.fit_transform(profiles_array)
            
            # Perform clustering
            n_clusters = min(3, len(model_profiles))  # Max 3 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_profiles)
            
            # Group models by cluster
            for i, cluster_id in enumerate(cluster_labels):
                cluster_name = f"Performance_Cluster_{cluster_id + 1}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(model_names[i])
            
        except Exception as e:
            logger.warning(f"Error in performance clustering: {e}")
        
        return clusters
    
    def _identify_variance_drivers(self, 
                                 data: pd.DataFrame, 
                                 dimension_variances: Dict[VarianceDimension, float]) -> List[Dict[str, Any]]:
        """Identify key drivers of variance"""
        drivers = []
        
        # Sort dimensions by variance (highest first)
        sorted_dimensions = sorted(dimension_variances.items(), key=lambda x: x[1], reverse=True)
        
        for dimension, variance_score in sorted_dimensions[:3]:  # Top 3 drivers
            driver = {
                'dimension': dimension.value,
                'variance_score': variance_score,
                'impact_level': 'high' if variance_score > 0.6 else 'medium' if variance_score > 0.3 else 'low',
                'description': self._get_driver_description(dimension, variance_score)
            }
            drivers.append(driver)
        
        return drivers
    
    def _get_driver_description(self, dimension: VarianceDimension, variance_score: float) -> str:
        """Get description for variance driver"""
        descriptions = {
            VarianceDimension.MODEL: f"Model selection drives {variance_score:.1%} of performance variance",
            VarianceDimension.DIFFICULTY_LEVEL: f"Document difficulty contributes {variance_score:.1%} to performance variation",
            VarianceDimension.DOCUMENT_TYPE: f"Document type differences account for {variance_score:.1%} of variance",
            VarianceDimension.CONTENT_DOMAIN: f"Content domain specificity causes {variance_score:.1%} performance variation"
        }
        
        return descriptions.get(dimension, f"{dimension.value} contributes {variance_score:.1%} to overall variance")
    
    def _identify_optimization_opportunities(self, 
                                           data: pd.DataFrame,
                                           dimension_variances: Dict[VarianceDimension, float],
                                           interaction_effects: Dict[Tuple[VarianceDimension, VarianceDimension], float]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on variance analysis"""
        opportunities = []
        
        # High variance dimensions = optimization opportunities
        for dimension, variance in dimension_variances.items():
            if variance > 0.5:
                opportunity = {
                    'type': 'variance_reduction',
                    'dimension': dimension.value,
                    'potential_improvement': f"{(1 - variance) * 100:.1f}%",
                    'priority': 'high' if variance > 0.7 else 'medium',
                    'action': self._get_optimization_action(dimension, variance)
                }
                opportunities.append(opportunity)
        
        # Strong interaction effects = opportunity for specialized approaches
        for (dim1, dim2), effect in interaction_effects.items():
            if effect > 0.4:
                opportunity = {
                    'type': 'interaction_optimization',
                    'dimensions': f"{dim1.value} x {dim2.value}",
                    'effect_size': f"{effect:.1%}",
                    'priority': 'medium',
                    'action': f"Develop specialized strategies for {dim1.value}-{dim2.value} combinations"
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _get_optimization_action(self, dimension: VarianceDimension, variance: float) -> str:
        """Get specific optimization action for dimension"""
        actions = {
            VarianceDimension.MODEL: "Standardize on top-performing models or implement ensemble methods",
            VarianceDimension.DIFFICULTY_LEVEL: "Implement adaptive processing based on document difficulty",
            VarianceDimension.DOCUMENT_TYPE: "Develop type-specific processing pipelines",
            VarianceDimension.CONTENT_DOMAIN: "Create domain-specialized models or preprocessing"
        }
        
        return actions.get(dimension, f"Investigate and address {dimension.value} variability")
    
    def _assess_variance_risks(self, 
                             dimension_variances: Dict[VarianceDimension, float],
                             overall_variance: float) -> Dict[str, Any]:
        """Assess risks associated with current variance levels"""
        
        # Risk levels
        risk_level = 'low'
        if overall_variance > 0.7:
            risk_level = 'high'
        elif overall_variance > 0.4:
            risk_level = 'medium'
        
        # Specific risks
        risks = []
        
        if overall_variance > 0.6:
            risks.append("Unpredictable performance in production environments")
        
        if VarianceDimension.MODEL in dimension_variances and dimension_variances[VarianceDimension.MODEL] > 0.5:
            risks.append("Model selection significantly impacts outcomes")
        
        if VarianceDimension.DIFFICULTY_LEVEL in dimension_variances and dimension_variances[VarianceDimension.DIFFICULTY_LEVEL] > 0.6:
            risks.append("Performance degrades significantly on challenging documents")
        
        # Mitigation strategies
        mitigations = []
        
        if risk_level == 'high':
            mitigations.extend([
                "Implement robust validation across all conditions",
                "Develop fallback strategies for poor-performing scenarios",
                "Monitor performance continuously in production"
            ])
        
        if VarianceDimension.MODEL in dimension_variances and dimension_variances[VarianceDimension.MODEL] > 0.5:
            mitigations.append("Standardize model selection criteria")
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': overall_variance,
            'specific_risks': risks,
            'mitigation_strategies': mitigations,
            'monitoring_recommendations': [
                "Track performance variance in real-time",
                "Set up alerts for unusual variance patterns",
                "Regular variance analysis reporting"
            ]
        }
    
    def _create_empty_result(self, dimension: VarianceDimension, metric: VarianceMetric, reason: str) -> VarianceResult:
        """Create empty result for error cases"""
        return VarianceResult(
            dimension=dimension,
            metric=metric,
            variance_score=0.5,
            stability_score=0.5,
            statistical_significance=1.0,
            effect_size=0.0,
            outliers=[],
            recommendations=[f"Unable to analyze {metric.value} variance across {dimension.value}: {reason}"],
            detailed_stats={'error': reason}
        )

# Utility functions for variance analysis

def calculate_confidence_intervals(data: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence intervals for a data series"""
    try:
        n = len(data)
        mean = data.mean()
        std_err = stats.sem(data)
        
        # Use t-distribution for small samples
        if n < 30:
            t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        else:
            t_val = stats.norm.ppf((1 + confidence_level) / 2)
        
        margin_error = t_val * std_err
        
        return (mean - margin_error, mean + margin_error)
        
    except Exception as e:
        logger.warning(f"Error calculating confidence intervals: {e}")
        return (data.mean(), data.mean())

def bootstrap_variance_estimate(data: pd.Series, n_bootstrap: int = 1000) -> Dict[str, float]:
    """Bootstrap estimate of variance statistics"""
    try:
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = data.sample(len(data), replace=True)
            bootstrap_vars.append(bootstrap_sample.var())
        
        bootstrap_vars = np.array(bootstrap_vars)
        
        return {
            'variance_estimate': np.mean(bootstrap_vars),
            'variance_ci_lower': np.percentile(bootstrap_vars, 2.5),
            'variance_ci_upper': np.percentile(bootstrap_vars, 97.5),
            'variance_std': np.std(bootstrap_vars)
        }
        
    except Exception as e:
        logger.warning(f"Error in bootstrap variance estimation: {e}")
        return {'variance_estimate': data.var(), 'variance_ci_lower': 0, 'variance_ci_upper': 0, 'variance_std': 0}

# Global analyzer instance
variance_analyzer = VarianceAnalyzer()
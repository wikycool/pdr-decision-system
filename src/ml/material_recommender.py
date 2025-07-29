#!/usr/bin/env python3
"""
AI/ML Material and Technology Recommender
Designed for enterprise PDR requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import logging
from datetime import datetime

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available")

class MaterialTechnologyRecommender:
    """
    AI/ML system for recommending materials and technologies based on requirements.
    Provides rough picture of product, materials, latest technology, and cost details.
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_material_database()
        self.setup_technology_database()
        self.setup_cost_database()
        
        if ML_AVAILABLE:
            self.setup_ml_models()
    
    def setup_logging(self):
        """Setup logging configuration."""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('material_recommender.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
            self.logger = None
    
    def setup_material_database(self):
        """Setup material database with properties and applications."""
        self.materials_db = {
            'aluminum': {
                'name': 'Aluminum',
                'properties': ['lightweight', 'corrosion_resistant', 'conductive', 'malleable'],
                'applications': ['electronics', 'automotive', 'aerospace', 'construction'],
                'cost_per_kg': 2.5,
                'strength': 'medium',
                'weight': 'light',
                'corrosion_resistance': 'high',
                'conductivity': 'high',
                'sustainability': 'high'
            },
            'stainless_steel': {
                'name': 'Stainless Steel',
                'properties': ['strong', 'corrosion_resistant', 'durable', 'hygienic'],
                'applications': ['medical', 'food_processing', 'marine', 'chemical'],
                'cost_per_kg': 8.0,
                'strength': 'high',
                'weight': 'medium',
                'corrosion_resistance': 'very_high',
                'conductivity': 'low',
                'sustainability': 'medium'
            },
            'carbon_steel': {
                'name': 'Carbon Steel',
                'properties': ['strong', 'cost_effective', 'weldable', 'versatile'],
                'applications': ['construction', 'automotive', 'machinery', 'piping'],
                'cost_per_kg': 1.2,
                'strength': 'high',
                'weight': 'heavy',
                'corrosion_resistance': 'low',
                'conductivity': 'medium',
                'sustainability': 'medium'
            },
            'titanium': {
                'name': 'Titanium',
                'properties': ['lightweight', 'strong', 'corrosion_resistant', 'biocompatible'],
                'applications': ['aerospace', 'medical', 'marine', 'sports'],
                'cost_per_kg': 45.0,
                'strength': 'very_high',
                'weight': 'light',
                'corrosion_resistance': 'very_high',
                'conductivity': 'low',
                'sustainability': 'high'
            },
            'plastic_abs': {
                'name': 'ABS Plastic',
                'properties': ['lightweight', 'impact_resistant', 'moldable', 'cost_effective'],
                'applications': ['consumer_goods', 'automotive', 'electronics', 'toys'],
                'cost_per_kg': 2.0,
                'strength': 'low',
                'weight': 'very_light',
                'corrosion_resistance': 'high',
                'conductivity': 'very_low',
                'sustainability': 'low'
            },
            'composite_carbon_fiber': {
                'name': 'Carbon Fiber Composite',
                'properties': ['lightweight', 'very_strong', 'stiff', 'corrosion_resistant'],
                'applications': ['aerospace', 'sports', 'automotive', 'marine'],
                'cost_per_kg': 120.0,
                'strength': 'very_high',
                'weight': 'very_light',
                'corrosion_resistance': 'very_high',
                'conductivity': 'low',
                'sustainability': 'medium'
            }
        }
    
    def setup_technology_database(self):
        """Setup latest technology database."""
        self.technology_db = {
            'additive_manufacturing': {
                'name': '3D Printing / Additive Manufacturing',
                'description': 'Layer-by-layer manufacturing for complex geometries',
                'applications': ['prototyping', 'custom_parts', 'complex_structures'],
                'cost_factor': 1.5,
                'lead_time': 'fast',
                'complexity': 'high',
                'sustainability': 'high'
            },
            'cnc_machining': {
                'name': 'CNC Machining',
                'description': 'Computer-controlled precision machining',
                'applications': ['precision_parts', 'mass_production', 'prototyping'],
                'cost_factor': 1.0,
                'lead_time': 'medium',
                'complexity': 'medium',
                'sustainability': 'medium'
            },
            'injection_molding': {
                'name': 'Injection Molding',
                'description': 'High-volume plastic manufacturing',
                'applications': ['mass_production', 'consumer_goods', 'automotive'],
                'cost_factor': 0.8,
                'lead_time': 'slow',
                'complexity': 'low',
                'sustainability': 'low'
            },
            'laser_cutting': {
                'name': 'Laser Cutting',
                'description': 'Precision cutting with laser technology',
                'applications': ['sheet_metal', 'prototyping', 'custom_parts'],
                'cost_factor': 1.2,
                'lead_time': 'fast',
                'complexity': 'low',
                'sustainability': 'medium'
            },
            'robotic_welding': {
                'name': 'Robotic Welding',
                'description': 'Automated welding for consistency',
                'applications': ['automotive', 'construction', 'manufacturing'],
                'cost_factor': 1.1,
                'lead_time': 'medium',
                'complexity': 'medium',
                'sustainability': 'medium'
            }
        }
    
    def setup_cost_database(self):
        """Setup cost analysis database."""
        self.cost_db = {
            'material_costs': {
                'aluminum': 2.5,
                'stainless_steel': 8.0,
                'carbon_steel': 1.2,
                'titanium': 45.0,
                'plastic_abs': 2.0,
                'composite_carbon_fiber': 120.0
            },
            'manufacturing_costs': {
                'additive_manufacturing': 1.5,
                'cnc_machining': 1.0,
                'injection_molding': 0.8,
                'laser_cutting': 1.2,
                'robotic_welding': 1.1
            },
            'labor_costs': {
                'high_skill': 50.0,  # per hour
                'medium_skill': 30.0,
                'low_skill': 15.0
            }
        }
    
    def setup_ml_models(self):
        """Setup ML models for recommendations."""
        try:
            # Create training data from material database
            training_data = []
            for material_id, material in self.materials_db.items():
                features = [
                    material['strength'] == 'high',
                    material['weight'] == 'light',
                    material['corrosion_resistance'] == 'high',
                    material['cost_per_kg'] < 10,
                    'electronics' in material['applications'],
                    'automotive' in material['applications']
                ]
                training_data.append({
                    'material_id': material_id,
                    'features': features,
                    'cost': material['cost_per_kg']
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            # Prepare features
            X = np.array([row['features'] for row in training_data])
            y_material = [row['material_id'] for row in training_data]
            y_cost = [row['cost'] for row in training_data]
            
            # Train models
            self.material_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.cost_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.material_classifier.fit(X, y_material)
            self.cost_regressor.fit(X, y_cost)
            
            print("✅ ML models trained successfully")
            
        except Exception as e:
            print(f"Error setting up ML models: {e}")
            self.material_classifier = None
            self.cost_regressor = None
    
    def analyze_requirements(self, requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze requirements and extract key features for recommendation."""
        
        analysis = {
            'strength_required': False,
            'lightweight_required': False,
            'corrosion_resistance_required': False,
            'cost_sensitive': False,
            'electronics_application': False,
            'automotive_application': False,
            'medical_application': False,
            'aerospace_application': False
        }
        
        # Analyze requirements text
        all_text = ' '.join([' '.join(reqs) for reqs in requirements.values()]).lower()
        
        # Check for strength requirements
        strength_keywords = ['strong', 'strength', 'load', 'stress', 'force', 'durable']
        if any(keyword in all_text for keyword in strength_keywords):
            analysis['strength_required'] = True
        
        # Check for lightweight requirements
        weight_keywords = ['light', 'lightweight', 'weight', 'mass', 'density']
        if any(keyword in all_text for keyword in weight_keywords):
            analysis['lightweight_required'] = True
        
        # Check for corrosion resistance
        corrosion_keywords = ['corrosion', 'rust', 'weather', 'environment', 'exposure']
        if any(keyword in all_text for keyword in corrosion_keywords):
            analysis['corrosion_resistance_required'] = True
        
        # Check for cost sensitivity
        cost_keywords = ['cost', 'budget', 'economic', 'cheap', 'affordable', 'price']
        if any(keyword in all_text for keyword in cost_keywords):
            analysis['cost_sensitive'] = True
        
        # Check for application types
        if 'electronics' in all_text or 'electrical' in all_text:
            analysis['electronics_application'] = True
        if 'automotive' in all_text or 'vehicle' in all_text:
            analysis['automotive_application'] = True
        if 'medical' in all_text or 'biocompatible' in all_text:
            analysis['medical_application'] = True
        if 'aerospace' in all_text or 'aircraft' in all_text:
            analysis['aerospace_application'] = True
        
        return analysis
    
    def recommend_materials(self, requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend materials based on requirements analysis."""
        
        recommendations = []
        
        # Convert analysis to feature vector
        features = [
            requirements_analysis['strength_required'],
            requirements_analysis['lightweight_required'],
            requirements_analysis['corrosion_resistance_required'],
            not requirements_analysis['cost_sensitive'],  # Invert cost sensitivity
            requirements_analysis['electronics_application'],
            requirements_analysis['automotive_application']
        ]
        
        # Use ML model if available
        if ML_AVAILABLE and self.material_classifier:
            try:
                # Get ML predictions
                predicted_material = self.material_classifier.predict([features])[0]
                predicted_cost = self.cost_regressor.predict([features])[0]
                
                # Get top 3 recommendations
                material_scores = []
                for material_id, material in self.materials_db.items():
                    score = 0
                    
                    # Score based on requirements
                    if requirements_analysis['strength_required'] and material['strength'] in ['high', 'very_high']:
                        score += 2
                    if requirements_analysis['lightweight_required'] and material['weight'] in ['light', 'very_light']:
                        score += 2
                    if requirements_analysis['corrosion_resistance_required'] and material['corrosion_resistance'] in ['high', 'very_high']:
                        score += 2
                    if requirements_analysis['cost_sensitive'] and material['cost_per_kg'] < 10:
                        score += 2
                    
                    # Score based on applications
                    if requirements_analysis['electronics_application'] and 'electronics' in material['applications']:
                        score += 1
                    if requirements_analysis['automotive_application'] and 'automotive' in material['applications']:
                        score += 1
                    if requirements_analysis['medical_application'] and 'medical' in material['applications']:
                        score += 1
                    if requirements_analysis['aerospace_application'] and 'aerospace' in material['applications']:
                        score += 1
                    
                    material_scores.append((material_id, score))
                
                # Sort by score and get top 3
                material_scores.sort(key=lambda x: x[1], reverse=True)
                top_materials = material_scores[:3]
                
                for material_id, score in top_materials:
                    material = self.materials_db[material_id]
                    recommendations.append({
                        'material_id': material_id,
                        'name': material['name'],
                        'score': score,
                        'cost_per_kg': material['cost_per_kg'],
                        'properties': material['properties'],
                        'applications': material['applications'],
                        'recommendation_reason': self._get_recommendation_reason(material, requirements_analysis)
                    })
                
            except Exception as e:
                print(f"Error in ML recommendation: {e}")
                # Fallback to rule-based recommendation
                recommendations = self._rule_based_material_recommendation(requirements_analysis)
        else:
            # Rule-based recommendation
            recommendations = self._rule_based_material_recommendation(requirements_analysis)
        
        return recommendations
    
    def _rule_based_material_recommendation(self, requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rule-based material recommendation."""
        recommendations = []
        
        for material_id, material in self.materials_db.items():
            score = 0
            reasons = []
            
            # Score based on requirements
            if requirements_analysis['strength_required'] and material['strength'] in ['high', 'very_high']:
                score += 2
                reasons.append("High strength requirement met")
            
            if requirements_analysis['lightweight_required'] and material['weight'] in ['light', 'very_light']:
                score += 2
                reasons.append("Lightweight requirement met")
            
            if requirements_analysis['corrosion_resistance_required'] and material['corrosion_resistance'] in ['high', 'very_high']:
                score += 2
                reasons.append("Corrosion resistance requirement met")
            
            if requirements_analysis['cost_sensitive'] and material['cost_per_kg'] < 10:
                score += 2
                reasons.append("Cost-effective option")
            
            # Score based on applications
            if requirements_analysis['electronics_application'] and 'electronics' in material['applications']:
                score += 1
                reasons.append("Suitable for electronics applications")
            
            if requirements_analysis['automotive_application'] and 'automotive' in material['applications']:
                score += 1
                reasons.append("Suitable for automotive applications")
            
            if requirements_analysis['medical_application'] and 'medical' in material['applications']:
                score += 1
                reasons.append("Suitable for medical applications")
            
            if requirements_analysis['aerospace_application'] and 'aerospace' in material['applications']:
                score += 1
                reasons.append("Suitable for aerospace applications")
            
            if score > 0:
                recommendations.append({
                    'material_id': material_id,
                    'name': material['name'],
                    'score': score,
                    'cost_per_kg': material['cost_per_kg'],
                    'properties': material['properties'],
                    'applications': material['applications'],
                    'recommendation_reason': reasons
                })
        
        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def _get_recommendation_reason(self, material: Dict[str, Any], requirements_analysis: Dict[str, Any]) -> List[str]:
        """Get reasons for material recommendation."""
        reasons = []
        
        if requirements_analysis['strength_required'] and material['strength'] in ['high', 'very_high']:
            reasons.append("Meets strength requirements")
        
        if requirements_analysis['lightweight_required'] and material['weight'] in ['light', 'very_light']:
            reasons.append("Meets lightweight requirements")
        
        if requirements_analysis['corrosion_resistance_required'] and material['corrosion_resistance'] in ['high', 'very_high']:
            reasons.append("Meets corrosion resistance requirements")
        
        if requirements_analysis['cost_sensitive'] and material['cost_per_kg'] < 10:
            reasons.append("Cost-effective option")
        
        return reasons
    
    def recommend_technologies(self, material_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recommend manufacturing technologies based on material recommendations."""
        
        recommendations = []
        
        for material_rec in material_recommendations:
            material_id = material_rec['material_id']
            
            for tech_id, technology in self.technology_db.items():
                score = 0
                reasons = []
                
                # Score based on material-technology compatibility
                if material_id in ['plastic_abs'] and tech_id == 'injection_molding':
                    score += 3
                    reasons.append("Ideal for plastic manufacturing")
                
                if material_id in ['aluminum', 'stainless_steel', 'carbon_steel'] and tech_id == 'cnc_machining':
                    score += 2
                    reasons.append("Suitable for metal machining")
                
                if material_id in ['aluminum', 'stainless_steel'] and tech_id == 'laser_cutting':
                    score += 2
                    reasons.append("Suitable for precision cutting")
                
                if tech_id == 'additive_manufacturing':
                    score += 1
                    reasons.append("Good for prototyping and complex geometries")
                
                if score > 0:
                    recommendations.append({
                        'technology_id': tech_id,
                        'name': technology['name'],
                        'description': technology['description'],
                        'score': score,
                        'cost_factor': technology['cost_factor'],
                        'lead_time': technology['lead_time'],
                        'material_compatibility': material_id,
                        'recommendation_reason': reasons
                    })
        
        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def estimate_costs(self, material_recommendations: List[Dict[str, Any]], 
                      technology_recommendations: List[Dict[str, Any]], 
                      estimated_weight_kg: float = 1.0) -> Dict[str, Any]:
        """Estimate costs for recommended materials and technologies."""
        
        cost_estimates = {
            'material_costs': [],
            'manufacturing_costs': [],
            'total_estimates': [],
            'cost_breakdown': {}
        }
        
        for material_rec in material_recommendations:
            material_cost = material_rec['cost_per_kg'] * estimated_weight_kg
            
            cost_estimates['material_costs'].append({
                'material': material_rec['name'],
                'cost_per_kg': material_rec['cost_per_kg'],
                'estimated_cost': material_cost
            })
            
            # Find compatible technology
            for tech_rec in technology_recommendations:
                if tech_rec['material_compatibility'] == material_rec['material_id']:
                    manufacturing_cost = material_cost * tech_rec['cost_factor']
                    total_cost = material_cost + manufacturing_cost
                    
                    cost_estimates['manufacturing_costs'].append({
                        'technology': tech_rec['name'],
                        'cost_factor': tech_rec['cost_factor'],
                        'manufacturing_cost': manufacturing_cost
                    })
                    
                    cost_estimates['total_estimates'].append({
                        'material': material_rec['name'],
                        'technology': tech_rec['name'],
                        'material_cost': material_cost,
                        'manufacturing_cost': manufacturing_cost,
                        'total_cost': total_cost,
                        'cost_per_unit': total_cost
                    })
        
        # Generate cost breakdown
        if cost_estimates['total_estimates']:
            min_cost = min(est['total_cost'] for est in cost_estimates['total_estimates'])
            max_cost = max(est['total_cost'] for est in cost_estimates['total_estimates'])
            avg_cost = sum(est['total_cost'] for est in cost_estimates['total_estimates']) / len(cost_estimates['total_estimates'])
            
            cost_estimates['cost_breakdown'] = {
                'cost_range': {'min': min_cost, 'max': max_cost, 'average': avg_cost},
                'recommendations': [
                    "Consider bulk purchasing for cost reduction",
                    "Evaluate alternative materials for cost optimization",
                    "Compare different manufacturing technologies"
                ]
            }
        
        return cost_estimates
    
    def generate_comprehensive_report(self, requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate comprehensive AI/ML report for PDR."""
        
        print("🔍 Analyzing requirements for AI/ML recommendations...")
        
        # Analyze requirements
        requirements_analysis = self.analyze_requirements(requirements)
        
        # Get material recommendations
        material_recommendations = self.recommend_materials(requirements_analysis)
        
        # Get technology recommendations
        technology_recommendations = self.recommend_technologies(material_recommendations)
        
        # Estimate costs
        cost_estimates = self.estimate_costs(material_recommendations, technology_recommendations)
        
        # Generate comprehensive report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'requirements_analysis': requirements_analysis,
            'material_recommendations': material_recommendations,
            'technology_recommendations': technology_recommendations,
            'cost_estimates': cost_estimates,
            'ai_ml_insights': self._generate_ai_insights(requirements_analysis, material_recommendations, technology_recommendations),
            'recommendations_summary': self._generate_recommendations_summary(material_recommendations, technology_recommendations, cost_estimates)
        }
        
        print(f"✅ AI/ML analysis complete. Generated recommendations for {len(material_recommendations)} materials and {len(technology_recommendations)} technologies")
        return report
    
    def _generate_ai_insights(self, requirements_analysis: Dict[str, Any], 
                             material_recommendations: List[Dict[str, Any]], 
                             technology_recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate AI insights from analysis."""
        
        insights = []
        
        # Material insights
        if material_recommendations:
            top_material = material_recommendations[0]
            insights.append(f"Top recommended material: {top_material['name']} (Score: {top_material['score']})")
            
            if top_material['cost_per_kg'] > 20:
                insights.append("High-cost material recommended - consider budget implications")
            elif top_material['cost_per_kg'] < 5:
                insights.append("Cost-effective material recommended - good for budget constraints")
        
        # Technology insights
        if technology_recommendations:
            top_technology = technology_recommendations[0]
            insights.append(f"Recommended manufacturing: {top_technology['name']}")
            
            if top_technology['lead_time'] == 'fast':
                insights.append("Fast manufacturing process - good for quick delivery")
            elif top_technology['lead_time'] == 'slow':
                insights.append("Slow manufacturing process - plan for longer lead times")
        
        # Requirements insights
        if requirements_analysis['strength_required']:
            insights.append("Strength requirements detected - high-strength materials recommended")
        
        if requirements_analysis['lightweight_required']:
            insights.append("Lightweight requirements detected - lightweight materials recommended")
        
        if requirements_analysis['cost_sensitive']:
            insights.append("Cost sensitivity detected - budget-friendly options prioritized")
        
        return insights
    
    def _generate_recommendations_summary(self, material_recommendations: List[Dict[str, Any]], 
                                        technology_recommendations: List[Dict[str, Any]], 
                                        cost_estimates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of recommendations."""
        
        summary = {
            'top_material': material_recommendations[0] if material_recommendations else None,
            'top_technology': technology_recommendations[0] if technology_recommendations else None,
            'cost_range': cost_estimates.get('cost_breakdown', {}).get('cost_range', {}),
            'key_decisions': [
                "Select final material based on requirements and budget",
                "Choose manufacturing technology based on volume and complexity",
                "Consider lead times for project planning",
                "Evaluate cost vs. performance trade-offs"
            ]
        }
        
        return summary

def main():
    """Test the material and technology recommender."""
    recommender = MaterialTechnologyRecommender()
    
    # Test with sample requirements
    sample_requirements = {
        'compliance': ['Must meet ISO 9001 standards', 'ASTM certification required'],
        'technical': ['High strength required', 'Lightweight design needed', 'Corrosion resistance important'],
        'cost': ['Budget constraint of $500 per unit', 'Cost-effective solution preferred']
    }
    
    print("Testing Material and Technology Recommender...")
    report = recommender.generate_comprehensive_report(sample_requirements)
    
    print(f"\n📊 AI/ML Analysis Results:")
    print(f"Top Material: {report['material_recommendations'][0]['name']}")
    print(f"Top Technology: {report['technology_recommendations'][0]['name']}")
    print(f"Cost Range: ${report['cost_estimates']['cost_breakdown']['cost_range']['min']:.2f} - ${report['cost_estimates']['cost_breakdown']['cost_range']['max']:.2f}")
    
    print(f"\n🤖 AI Insights:")
    for insight in report['ai_ml_insights']:
        print(f"  - {insight}")

if __name__ == "__main__":
    main() 
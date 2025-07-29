#!/usr/bin/env python3
"""
Enhanced Pugh Matrix for Decision Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .pugh import pugh_matrix, load_criteria_from_yaml

class EnhancedPughMatrix:
    """
    Enhanced Pugh Matrix class for decision analysis.
    """
    
    def __init__(self):
        self.criteria = {}
        self.weights = {}
        self.alternatives = {}
        self.baseline = None
        self.ratings = {}
    
    def set_criteria(self, criteria: Dict[str, str], weights: Dict[str, float]):
        """Set criteria and weights."""
        self.criteria = criteria
        self.weights = weights
    
    def add_alternative(self, alt_id: str, description: str, specs: Dict[str, Any]):
        """Add an alternative to the matrix."""
        self.alternatives[alt_id] = {
            'description': description,
            'specs': specs
        }
    
    def set_baseline(self, baseline_id: str):
        """Set the baseline alternative."""
        self.baseline = baseline_id
    
    def rate_alternative(self, alt_id: str, criteria_id: str, rating: int, comment: str = ""):
        """Rate an alternative against a criteria."""
        if alt_id not in self.ratings:
            self.ratings[alt_id] = {}
        self.ratings[alt_id][criteria_id] = {
            'rating': rating,
            'comment': comment
        }
    
    def calculate_scores(self) -> List[Dict[str, Any]]:
        """Calculate scores for all alternatives."""
        if not self.criteria or not self.alternatives:
            return []
        
        results = []
        
        for alt_id, alt_data in self.alternatives.items():
            scores = []
            total_weight = 0
            final_score = 0
            
            for criteria_id, weight in self.weights.items():
                rating = 0
                if alt_id in self.ratings and criteria_id in self.ratings[alt_id]:
                    rating = self.ratings[alt_id][criteria_id]['rating']
                
                scores.append(rating)
                final_score += rating * weight
                total_weight += weight
            
            # Normalize score
            normalized_score = final_score / total_weight if total_weight > 0 else 0
            
            results.append({
                'alternative_id': alt_id,
                'description': alt_data['description'],
                'final_score': normalized_score,
                'total_weight': total_weight,
                'scores': scores
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed report."""
        scores = self.calculate_scores()
        
        if not scores:
            return {'summary': {'best_alternative': None}}
        
        best_alternative = scores[0]
        
        return {
            'summary': {
                'best_alternative': best_alternative['alternative_id'],
                'best_score': best_alternative['final_score'],
                'total_alternatives': len(scores)
            },
            'detailed_results': scores
        }
    
    def export_to_csv(self, filename: str):
        """Export results to CSV."""
        scores = self.calculate_scores()
        if scores:
            df = pd.DataFrame(scores)
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}") 
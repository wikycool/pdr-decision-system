import pandas as pd
import numpy as np
import yaml
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization disabled.")

class EnhancedPughMatrix:
    """
    Enhanced Pugh Matrix for PDR (Preliminary Design Review) decision making.
    Supports multiple criteria, weightings, and generates comprehensive reports.
    """
    
    def __init__(self):
        self.criteria = {}
        self.alternatives = {}
        self.ratings = {}
        self.weights = {}
        self.baseline = None
        self.results = []
        
    def load_criteria_from_yaml(self, yaml_file: str) -> None:
        """Load criteria and weights from YAML configuration."""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'requirements' not in data:
                raise ValueError(f"Invalid YAML structure in {yaml_file}")
            
            self.criteria = {item['id']: item['description'] for item in data['requirements']}
            self.weights = {item['id']: item['weight'] for item in data['requirements']}
            
            print(f"Loaded {len(self.criteria)} criteria from {yaml_file}")
        except FileNotFoundError:
            print(f"Error: Configuration file {yaml_file} not found")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
        except Exception as e:
            print(f"Error loading criteria: {e}")
    
    def add_alternative(self, alt_id: str, description: str, specifications: Dict[str, Any]) -> None:
        """Add an alternative to the decision matrix."""
        if not alt_id or not description:
            print("Error: Alternative ID and description are required")
            return
            
        self.alternatives[alt_id] = {
            'description': description,
            'specifications': specifications or {},
            'added_timestamp': datetime.now().isoformat()
        }
        print(f"Added alternative: {alt_id} - {description}")
    
    def set_baseline(self, baseline_id: str) -> None:
        """Set the baseline alternative for comparison."""
        if not baseline_id:
            print("Error: Baseline ID is required")
            return
            
        if baseline_id in self.alternatives:
            self.baseline = baseline_id
            print(f"Baseline set to: {baseline_id}")
        else:
            print(f"Error: Alternative {baseline_id} not found")
    
    def rate_alternative(self, alt_id: str, criteria_id: str, rating: int, comments: str = "") -> None:
        """
        Rate an alternative against a criterion.
        Rating: -2 (much worse), -1 (worse), 0 (same), +1 (better), +2 (much better)
        """
        if alt_id not in self.alternatives:
            print(f"Error: Alternative {alt_id} not found")
            return
        
        if criteria_id not in self.criteria:
            print(f"Error: Criterion {criteria_id} not found")
            return
        
        if rating not in [-2, -1, 0, 1, 2]:
            print("Error: Rating must be -2, -1, 0, 1, or 2")
            return
        
        if alt_id not in self.ratings:
            self.ratings[alt_id] = {}
        
        self.ratings[alt_id][criteria_id] = {
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_scores(self) -> pd.DataFrame:
        """Calculate weighted scores for all alternatives."""
        if not self.baseline:
            print("Error: No baseline set")
            return pd.DataFrame()
        
        if not self.alternatives:
            print("Error: No alternatives defined")
            return pd.DataFrame()
        
        # Create results DataFrame
        results_data = []
        
        for alt_id in self.alternatives:
            if alt_id == self.baseline:
                continue
            
            alt_scores = []
            total_weighted_score = 0
            total_weight = 0
            
            for criteria_id in self.criteria:
                if alt_id in self.ratings and criteria_id in self.ratings[alt_id]:
                    rating = self.ratings[alt_id][criteria_id]['rating']
                    weight = self.weights[criteria_id]
                    weighted_score = rating * weight
                    
                    alt_scores.append({
                        'criteria': criteria_id,
                        'rating': rating,
                        'weight': weight,
                        'weighted_score': weighted_score
                    })
                    
                    total_weighted_score += weighted_score
                    total_weight += weight
                else:
                    alt_scores.append({
                        'criteria': criteria_id,
                        'rating': 0,
                        'weight': self.weights[criteria_id],
                        'weighted_score': 0
                    })
            
            # Calculate final score
            final_score = total_weighted_score if total_weight > 0 else 0
            
            results_data.append({
                'alternative_id': alt_id,
                'description': self.alternatives[alt_id]['description'],
                'final_score': final_score,
                'total_weight': total_weight,
                'scores': alt_scores
            })
        
        # Sort by final score (descending)
        results_data.sort(key=lambda x: x['final_score'], reverse=True)
        
        self.results = results_data
        return pd.DataFrame(results_data)
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed Pugh matrix report."""
        if not self.results:
            self.calculate_scores()
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'baseline': self.baseline,
            'total_alternatives': len(self.alternatives),
            'total_criteria': len(self.criteria),
            'summary': {
                'best_alternative': self.results[0]['alternative_id'] if self.results else None,
                'best_score': self.results[0]['final_score'] if self.results else 0,
                'score_range': {
                    'min': min([r['final_score'] for r in self.results]) if self.results else 0,
                    'max': max([r['final_score'] for r in self.results]) if self.results else 0
                }
            },
            'detailed_results': self.results,
            'criteria_summary': self._generate_criteria_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_criteria_summary(self) -> Dict[str, Any]:
        """Generate summary of criteria performance."""
        criteria_summary = {}
        
        for criteria_id in self.criteria:
            ratings = []
            for alt_id in self.alternatives:
                if alt_id != self.baseline and alt_id in self.ratings:
                    if criteria_id in self.ratings[alt_id]:
                        ratings.append(self.ratings[alt_id][criteria_id]['rating'])
            
            if ratings:
                criteria_summary[criteria_id] = {
                    'description': self.criteria[criteria_id],
                    'weight': self.weights[criteria_id],
                    'avg_rating': float(np.mean(ratings)),
                    'std_rating': float(np.std(ratings)),
                    'total_ratings': len(ratings)
                }
        
        return criteria_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Best alternative recommendation
        best_alt = self.results[0]
        recommendations.append(f"Recommended alternative: {best_alt['alternative_id']} (Score: {best_alt['final_score']})")
        
        # Score analysis
        scores = [r['final_score'] for r in self.results]
        if max(scores) - min(scores) < 10:
            recommendations.append("Warning: Close scores suggest need for additional analysis")
        
        # Criteria analysis
        for criteria_id, summary in self._generate_criteria_summary().items():
            if summary['avg_rating'] < 0:
                recommendations.append(f"Focus on improving {criteria_id} performance across alternatives")
        
        return recommendations
    
    def create_visualization(self, save_path: Optional[str] = None) -> None:
        """Create visualization of the Pugh matrix results."""
        if not PLOTTING_AVAILABLE:
            print("Warning: Visualization not available (matplotlib/seaborn not installed)")
            return
            
        if not self.results:
            self.calculate_scores()
        
        if not self.results:
            print("No results to visualize")
            return
        
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar chart of final scores
            alternatives = [r['alternative_id'] for r in self.results]
            scores = [r['final_score'] for r in self.results]
            
            bars = ax1.bar(alternatives, scores, color='skyblue')
            ax1.set_title('Pugh Matrix - Final Scores')
            ax1.set_ylabel('Weighted Score')
            ax1.set_xlabel('Alternatives')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.1f}', ha='center', va='bottom')
            
            # Heatmap of ratings
            if self.results and self.results[0]['scores']:
                criteria = [score['criteria'] for score in self.results[0]['scores']]
                ratings_matrix = []
                
                for result in self.results:
                    row = [score['rating'] for score in result['scores']]
                    ratings_matrix.append(row)
                
                im = ax2.imshow(ratings_matrix, cmap='RdYlGn', aspect='auto')
                ax2.set_xticks(range(len(criteria)))
                ax2.set_yticks(range(len(alternatives)))
                ax2.set_xticklabels(criteria, rotation=45, ha='right')
                ax2.set_yticklabels(alternatives)
                ax2.set_title('Ratings Heatmap')
                
                # Add colorbar
                plt.colorbar(im, ax=ax2, label='Rating (-2 to +2)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.show()
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def export_to_csv(self, file_path: str) -> None:
        """Export results to CSV file."""
        if not self.results:
            self.calculate_scores()
        
        if not self.results:
            print("No results to export")
            return
        
        try:
            # Create detailed CSV
            rows = []
            for result in self.results:
                for score in result['scores']:
                    rows.append({
                        'alternative_id': result['alternative_id'],
                        'description': result['description'],
                        'criteria': score['criteria'],
                        'rating': score['rating'],
                        'weight': score['weight'],
                        'weighted_score': score['weighted_score'],
                        'final_score': result['final_score']
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
            print(f"Results exported to {file_path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def save_report(self, file_path: str) -> None:
        """Save complete analysis report to JSON file."""
        try:
            report = self.generate_detailed_report()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Report saved to {file_path}")
        except Exception as e:
            print(f"Error saving report: {e}")

def main():
    """Example usage of Enhanced Pugh Matrix."""
    pugh = EnhancedPughMatrix()
    
    # Load criteria from config
    pugh.load_criteria_from_yaml('config/requirements.yml')
    
    # Add alternatives
    pugh.add_alternative('ALT_001', 'Aluminum Enclosure', {
        'material': 'Aluminum',
        'cost': 150,
        'weight': 2.5,
        'corrosion_resistance': 'High'
    })
    
    pugh.add_alternative('ALT_002', 'Stainless Steel Enclosure', {
        'material': 'Stainless Steel',
        'cost': 300,
        'weight': 4.0,
        'corrosion_resistance': 'Very High'
    })
    
    pugh.add_alternative('ALT_003', 'Plastic Enclosure', {
        'material': 'ABS Plastic',
        'cost': 80,
        'weight': 1.5,
        'corrosion_resistance': 'Medium'
    })
    
    # Set baseline
    pugh.set_baseline('ALT_001')
    
    # Rate alternatives (example ratings)
    ratings = {
        'ALT_002': {'R1': 1, 'R2': 1, 'R3': -1},  # Better corrosion, better attenuation, heavier
        'ALT_003': {'R1': -1, 'R2': -1, 'R3': 1}   # Worse corrosion, worse attenuation, lighter
    }
    
    for alt_id, alt_ratings in ratings.items():
        for criteria_id, rating in alt_ratings.items():
            pugh.rate_alternative(alt_id, criteria_id, rating)
    
    # Calculate and display results
    results_df = pugh.calculate_scores()
    print("\nPugh Matrix Results:")
    print(results_df[['alternative_id', 'description', 'final_score']])
    
    # Generate report
    report = pugh.generate_detailed_report()
    print(f"\nBest alternative: {report['summary']['best_alternative']}")
    
    # Create visualization
    pugh.create_visualization('pugh_matrix_results.png')
    
    # Export results
    pugh.export_to_csv('pugh_matrix_results.csv')
    pugh.save_report('pugh_matrix_report.json')

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import yaml
from pathlib import Path

def pugh_matrix(criteria_dict: Dict[str, float], alternatives_dict: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate Pugh matrix scores for alternatives.
    
    Args:
        criteria_dict: Dictionary of criteria and their weights
        alternatives_dict: Dictionary of alternatives and their scores for each criterion
        
    Returns:
        DataFrame with Pugh matrix results
    """
    try:
        if not criteria_dict:
            print("Error: No criteria provided")
            return pd.DataFrame()
        
        if not alternatives_dict:
            print("Error: No alternatives provided")
            return pd.DataFrame()
        
        # Create results DataFrame
        results = []
        
        for alt_id, scores in alternatives_dict.items():
            if len(scores) != len(criteria_dict):
                print(f"Warning: Alternative {alt_id} has {len(scores)} scores but {len(criteria_dict)} criteria")
                continue
            
            # Calculate weighted score
            weighted_score = 0
            total_weight = 0
            
            for i, (criteria_id, weight) in enumerate(criteria_dict.items()):
                if i < len(scores):
                    weighted_score += scores[i] * weight
                    total_weight += weight
            
            # Normalize score
            final_score = weighted_score / total_weight if total_weight > 0 else 0
            
            results.append({
                'alternative_id': alt_id,
                'weighted_score': weighted_score,
                'final_score': final_score,
                'total_weight': total_weight
            })
        
        # Create DataFrame and sort by final score
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('final_score', ascending=False)
        
        return df_results
        
    except Exception as e:
        print(f"Error calculating Pugh matrix: {e}")
        return pd.DataFrame()

def load_criteria_from_yaml(yaml_file: str) -> Dict[str, float]:
    """
    Load criteria and weights from YAML file.
    
    Args:
        yaml_file: Path to YAML configuration file
        
    Returns:
        Dictionary of criteria and weights
    """
    try:
        if not Path(yaml_file).exists():
            print(f"Error: YAML file {yaml_file} not found")
            return {}
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data or 'requirements' not in data:
            print(f"Error: Invalid YAML structure in {yaml_file}")
            return {}
        
        criteria_dict = {}
        for item in data['requirements']:
            if 'id' in item and 'weight' in item:
                criteria_dict[item['id']] = float(item['weight'])
        
        print(f"Loaded {len(criteria_dict)} criteria from {yaml_file}")
        return criteria_dict
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_file}: {e}")
        return {}
    except Exception as e:
        print(f"Error loading criteria from YAML: {e}")
        return {}

def create_pugh_matrix_data(criteria_dict: Dict[str, float], 
                           alternatives: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Create data structures for Pugh matrix calculation.
    
    Args:
        criteria_dict: Dictionary of criteria and weights
        alternatives: List of alternative dictionaries with scores
        
    Returns:
        Tuple of (criteria_dict, alternatives_dict)
    """
    try:
        alternatives_dict = {}
        
        for alt in alternatives:
            if 'id' not in alt or 'scores' not in alt:
                print(f"Warning: Alternative missing 'id' or 'scores': {alt}")
                continue
            
            alt_id = alt['id']
            scores = alt['scores']
            
            # Ensure scores match criteria
            if len(scores) != len(criteria_dict):
                print(f"Warning: Alternative {alt_id} has {len(scores)} scores but {len(criteria_dict)} criteria")
                # Pad or truncate scores
                if len(scores) < len(criteria_dict):
                    scores.extend([0] * (len(criteria_dict) - len(scores)))
                else:
                    scores = scores[:len(criteria_dict)]
            
            alternatives_dict[alt_id] = scores
        
        return criteria_dict, alternatives_dict
        
    except Exception as e:
        print(f"Error creating Pugh matrix data: {e}")
        return {}, {}

def calculate_pugh_scores(criteria_dict: Dict[str, float], 
                         alternatives_dict: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate Pugh matrix scores with detailed breakdown.
    
    Args:
        criteria_dict: Dictionary of criteria and weights
        alternatives_dict: Dictionary of alternatives and their scores
        
    Returns:
        DataFrame with detailed Pugh matrix results
    """
    try:
        if not criteria_dict or not alternatives_dict:
            print("Error: Missing criteria or alternatives")
            return pd.DataFrame()
        
        detailed_results = []
        
        for alt_id, scores in alternatives_dict.items():
            alt_result = {
                'alternative_id': alt_id,
                'total_weighted_score': 0,
                'total_weight': 0
            }
            
            # Calculate scores for each criterion
            for i, (criteria_id, weight) in enumerate(criteria_dict.items()):
                score = scores[i] if i < len(scores) else 0
                weighted_score = score * weight
                
                alt_result[f'{criteria_id}_score'] = score
                alt_result[f'{criteria_id}_weighted'] = weighted_score
                alt_result['total_weighted_score'] += weighted_score
                alt_result['total_weight'] += weight
            
            # Calculate final score
            alt_result['final_score'] = (alt_result['total_weighted_score'] / 
                                       alt_result['total_weight'] if alt_result['total_weight'] > 0 else 0)
            
            detailed_results.append(alt_result)
        
        df_results = pd.DataFrame(detailed_results)
        if not df_results.empty:
            df_results = df_results.sort_values('final_score', ascending=False)
        
        return df_results
        
    except Exception as e:
        print(f"Error calculating detailed Pugh scores: {e}")
        return pd.DataFrame()

def validate_pugh_data(criteria_dict: Dict[str, float], 
                      alternatives_dict: Dict[str, List[float]]) -> bool:
    """
    Validate Pugh matrix input data.
    
    Args:
        criteria_dict: Dictionary of criteria and weights
        alternatives_dict: Dictionary of alternatives and their scores
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check criteria
        if not criteria_dict:
            print("Error: No criteria provided")
            return False
        
        for criteria_id, weight in criteria_dict.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                print(f"Error: Invalid weight for criteria {criteria_id}: {weight}")
                return False
        
        # Check alternatives
        if not alternatives_dict:
            print("Error: No alternatives provided")
            return False
        
        for alt_id, scores in alternatives_dict.items():
            if not isinstance(scores, list):
                print(f"Error: Scores for alternative {alt_id} must be a list")
                return False
            
            for score in scores:
                if not isinstance(score, (int, float)):
                    print(f"Error: Invalid score in alternative {alt_id}: {score}")
                    return False
        
        print("Pugh matrix data validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating Pugh matrix data: {e}")
        return False

def export_pugh_results(df_results: pd.DataFrame, output_file: str) -> None:
    """
    Export Pugh matrix results to CSV file.
    
    Args:
        df_results: DataFrame with Pugh matrix results
        output_file: Path to output CSV file
    """
    try:
        if df_results.empty:
            print("Warning: No results to export")
            return
        
        # Create directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        df_results.to_csv(output_file, index=False)
        print(f"Pugh matrix results exported to {output_file}")
        
    except Exception as e:
        print(f"Error exporting Pugh matrix results: {e}")

def main():
    """Example usage of Pugh matrix functions."""
    # Sample criteria and weights
    criteria_dict = {
        'R1': 10,  # Compliance Requirements
        'R2': 8,   # Technical Specifications
        'R3': 6    # Cost Considerations
    }
    
    # Sample alternatives with scores
    alternatives_dict = {
        'ALT_001': [1, 1, -1],   # Better compliance, better technical, higher cost
        'ALT_002': [1, 1, -1],   # Better compliance, better technical, higher cost
        'ALT_003': [-1, -1, 1]   # Worse compliance, worse technical, lower cost
    }
    
    # Validate data
    if validate_pugh_data(criteria_dict, alternatives_dict):
        # Calculate Pugh matrix
        results = pugh_matrix(criteria_dict, alternatives_dict)
        
        if not results.empty:
            print("\nPugh Matrix Results:")
            print(results)
            
            # Export results
            export_pugh_results(results, "pugh_matrix_results.csv")
        else:
            print("No results generated")
    else:
        print("Data validation failed")

if __name__ == "__main__":
    main()
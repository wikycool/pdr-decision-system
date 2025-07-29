#!/usr/bin/env python3
"""
Generate test output files with sample data
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def generate_test_csv():
    """Generate a sample CSV file with Pugh matrix results."""
    data = {
        'alternative_id': ['ALT_001', 'ALT_002', 'ALT_003'],
        'description': ['Aluminum Enclosure', 'Stainless Steel Enclosure', 'Plastic Enclosure'],
        'final_score': [15.5, 12.3, 8.7],
        'total_weight': [10.0, 10.0, 10.0],
        'cost_score': [2, 1, 2],
        'weight_score': [2, 1, 2],
        'corrosion_score': [1, 2, 0],
        'strength_score': [1, 2, 0],
        'manufacturability_score': [2, 1, 2]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('pdr_results.csv', index=False)
    print("✅ Generated pdr_results.csv with sample data")

def generate_test_json():
    """Generate a sample JSON report."""
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "baseline": "ALT_001",
        "total_alternatives": 3,
        "total_criteria": 5,
        "summary": {
            "best_alternative": "ALT_001",
            "best_score": 15.5,
            "score_range": {
                "min": 8.7,
                "max": 15.5
            }
        },
        "detailed_results": [
            {
                "alternative_id": "ALT_001",
                "description": "Aluminum Enclosure",
                "final_score": 15.5,
                "total_weight": 10.0,
                "scores": [2, 2, 1, 1, 2]
            },
            {
                "alternative_id": "ALT_002",
                "description": "Stainless Steel Enclosure",
                "final_score": 12.3,
                "total_weight": 10.0,
                "scores": [1, 1, 2, 2, 1]
            },
            {
                "alternative_id": "ALT_003",
                "description": "Plastic Enclosure",
                "final_score": 8.7,
                "total_weight": 10.0,
                "scores": [2, 2, 0, 0, 2]
            }
        ],
        "criteria_summary": {
            "cost": {"weight": 0.2, "description": "Cost effectiveness"},
            "weight": {"weight": 0.2, "description": "Lightweight design"},
            "corrosion": {"weight": 0.2, "description": "Corrosion resistance"},
            "strength": {"weight": 0.2, "description": "Structural strength"},
            "manufacturability": {"weight": 0.2, "description": "Ease of manufacturing"}
        },
        "recommendations": [
            "Recommended alternative: ALT_001 (Aluminum Enclosure) with score 15.5",
            "Aluminum provides the best balance of cost, weight, and manufacturability",
            "Consider stainless steel if higher corrosion resistance is required"
        ]
    }
    
    with open('pdr_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("✅ Generated pdr_report.json with sample data")

def generate_test_png():
    """Generate a sample PNG chart."""
    # Sample data
    alternatives = ['Aluminum', 'Stainless Steel', 'Plastic']
    scores = [15.5, 12.3, 8.7]
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(alternatives, scores, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.title('Pugh Matrix Results - Alternative Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Alternatives', fontsize=12)
    plt.ylabel('Final Score', fontsize=12)
    plt.ylim(0, max(scores) + 2)
    plt.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line at the baseline
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Baseline')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pdr_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated pdr_results.png with sample chart")

def main():
    """Generate all test output files."""
    print("🚀 Generating test output files...")
    
    try:
        generate_test_csv()
        generate_test_json()
        generate_test_png()
        
        print("\n🎉 All test files generated successfully!")
        print("📁 Files created:")
        print("   - pdr_results.csv (Pugh matrix results)")
        print("   - pdr_report.json (Detailed analysis report)")
        print("   - pdr_results.png (Visualization chart)")
        
        print("\n📖 How to open these files:")
        print("   - CSV: Open with Excel or Google Sheets")
        print("   - JSON: Open with any text editor (Notepad, VS Code)")
        print("   - PNG: Open with any image viewer")
        
    except Exception as e:
        print(f"❌ Error generating files: {e}")

if __name__ == "__main__":
    main() 
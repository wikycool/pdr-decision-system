import pandas as pd
import numpy as np
import yaml
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization disabled.")

# Import our modules
try:
    from analysis.document_analyzer import DocumentAnalyzer
    from decision.pugh_enhanced import EnhancedPughMatrix
    from reports.dar_report import generate_dar
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    DocumentAnalyzer = None
    EnhancedPughMatrix = None
    generate_dar = None

class PDRWorkflow:
    """
    Comprehensive PDR (Preliminary Design Review) workflow system.
    Orchestrates the entire process from document analysis to final concept selection.
    """
    
    def __init__(self, config_path: str = 'config/pdr_config.yml'):
        self.config_path = config_path
        self.document_analyzer = DocumentAnalyzer() if DocumentAnalyzer else None
        self.pugh_matrix = EnhancedPughMatrix() if EnhancedPughMatrix else None
        self.workflow_steps = []
        self.results = {}
        
        self.setup_logging()
        self.load_config()
    
    def setup_logging(self):
        """Setup logging configuration."""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('pdr_workflow.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
            self.logger = None
    
    def load_config(self):
        """Load PDR configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if self.logger:
                self.logger.info(f"Loaded PDR configuration from {self.config_path}")
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            else:
                print(f"Warning: Config file {self.config_path} not found, using defaults")
            self.config = self._get_default_config()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading config: {e}")
            else:
                print(f"Error loading config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default PDR configuration."""
        return {
            'workflow_steps': [
                'document_analysis',
                'requirements_extraction',
                'ideation',
                'brainstorming',
                'concept_selection',
                'final_approval'
            ],
            'criteria_weights': {
                'compliance': 0.3,
                'technical': 0.4,
                'cost': 0.2,
                'risk': 0.1
            },
            'output_formats': ['json', 'csv', 'pdf', 'png']
        }
    
    def step_1_document_analysis(self, document_paths: List[str]) -> Dict[str, Any]:
        """Step 1: Analyze documents and extract T&C, compliance standards."""
        if not self.document_analyzer:
            raise ValueError("DocumentAnalyzer not available")
            
        if self.logger:
            self.logger.info("Starting Step 1: Document Analysis")
        
        # Analyze documents
        analysis_results = self.document_analyzer.analyze_multiple_documents(document_paths)
        
        # Extract key insights
        insights = {
            'total_requirements': analysis_results['overall_summary']['total_requirements'],
            'compliance_requirements': len(analysis_results['combined_requirements']['compliance']),
            'technical_requirements': len(analysis_results['combined_requirements']['technical']),
            'cost_requirements': len(analysis_results['combined_requirements']['cost']),
            'key_recommendations': analysis_results['overall_summary']['recommendations']
        }
        
        self.results['document_analysis'] = analysis_results
        self.results['insights'] = insights
        
        if self.logger:
            self.logger.info(f"Document analysis completed. Found {insights['total_requirements']} requirements")
        return insights
    
    def step_2_requirements_extraction(self) -> Dict[str, Any]:
        """Step 2: Convert requirements into decision criteria."""
        if self.logger:
            self.logger.info("Starting Step 2: Requirements Extraction")
        
        if 'document_analysis' not in self.results:
            raise ValueError("Document analysis must be completed first")
        
        # Extract requirements from analysis
        requirements = self.results['document_analysis']['combined_requirements']
        
        # Convert to decision criteria
        criteria = {}
        weights = {}
        
        # Compliance criteria
        if requirements['compliance']:
            criteria['R1'] = "Compliance Requirements"
            weights['R1'] = 10  # High priority
        
        # Technical criteria
        if requirements['technical']:
            criteria['R2'] = "Technical Specifications"
            weights['R2'] = 8   # High priority
        
        # Cost criteria
        if requirements['cost']:
            criteria['R3'] = "Cost Considerations"
            weights['R3'] = 6   # Medium priority
        
        # Create requirements config
        requirements_config = {
            'requirements': [
                {'id': k, 'description': v, 'weight': weights[k], 'moscow': 'Must'}
                for k, v in criteria.items()
            ]
        }
        
        # Save to config file
        try:
            with open('config/requirements.yml', 'w', encoding='utf-8') as f:
                yaml.dump(requirements_config, f, default_flow_style=False)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving requirements config: {e}")
            else:
                print(f"Error saving requirements config: {e}")
        
        self.results['requirements_extraction'] = {
            'criteria': criteria,
            'weights': weights,
            'config_file': 'config/requirements.yml'
        }
        
        if self.logger:
            self.logger.info(f"Requirements extraction completed. Created {len(criteria)} criteria")
        return self.results['requirements_extraction']
    
    def step_3_ideation(self, alternatives_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Generate and evaluate alternatives."""
        if not self.pugh_matrix:
            raise ValueError("EnhancedPughMatrix not available")
            
        if self.logger:
            self.logger.info("Starting Step 3: Ideation")
        
        # Load criteria
        self.pugh_matrix.load_criteria_from_yaml('config/requirements.yml')
        
        # Add alternatives
        for alt_data in alternatives_data:
            self.pugh_matrix.add_alternative(
                alt_data['id'],
                alt_data['description'],
                alt_data['specifications']
            )
        
        # Set baseline (first alternative)
        baseline_id = None
        if alternatives_data:
            baseline_id = alternatives_data[0]['id']
            self.pugh_matrix.set_baseline(baseline_id)
        
        self.results['ideation'] = {
            'alternatives_count': len(alternatives_data),
            'baseline': baseline_id,
            'alternatives': alternatives_data
        }
        
        if self.logger:
            self.logger.info(f"Ideation completed. Added {len(alternatives_data)} alternatives")
        return self.results['ideation']
    
    def step_4_brainstorming(self, ratings_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Step 4: Cross-functional team brainstorming and rating."""
        if not self.pugh_matrix:
            raise ValueError("EnhancedPughMatrix not available")
            
        if self.logger:
            self.logger.info("Starting Step 4: Brainstorming")
        
        # Apply ratings from cross-functional team
        for alt_id, ratings in ratings_data.items():
            for criteria_id, rating in ratings.items():
                self.pugh_matrix.rate_alternative(alt_id, criteria_id, rating)
        
        # Calculate scores
        results_df = self.pugh_matrix.calculate_scores()
        
        # Generate detailed report
        report = self.pugh_matrix.generate_detailed_report()
        
        self.results['brainstorming'] = {
            'ratings_applied': len(ratings_data),
            'results_dataframe': results_df,
            'detailed_report': report,
            'best_alternative': report['summary']['best_alternative']
        }
        
        if self.logger:
            self.logger.info(f"Brainstorming completed. Best alternative: {report['summary']['best_alternative']}")
        return self.results['brainstorming']
    
    def step_5_concept_selection(self) -> Dict[str, Any]:
        """Step 5: Narrow down to final concept selection."""
        if self.logger:
            self.logger.info("Starting Step 5: Concept Selection")
        
        if 'brainstorming' not in self.results:
            raise ValueError("Brainstorming must be completed first")
        
        # Get top 3 alternatives
        results_df = self.results['brainstorming']['results_dataframe']
        top_3 = results_df.head(3)
        
        # Create final selection matrix
        final_selection = {
            'top_3_alternatives': top_3.to_dict('records'),
            'final_recommendation': top_3.iloc[0].to_dict() if not top_3.empty else {},
            'selection_criteria': list(self.pugh_matrix.criteria.keys()) if self.pugh_matrix else [],
            'selection_timestamp': datetime.now().isoformat()
        }
        
        self.results['concept_selection'] = final_selection
        
        if self.logger:
            self.logger.info(f"Concept selection completed. Final recommendation: {final_selection['final_recommendation'].get('alternative_id', 'None')}")
        return final_selection
    
    def step_6_final_approval(self, client_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Step 6: Client approval and final documentation."""
        if self.logger:
            self.logger.info("Starting Step 6: Final Approval")
        
        # Generate comprehensive PDR report
        pdr_report = self.generate_pdr_report()
        
        # Add client feedback if provided
        if client_feedback:
            pdr_report['client_feedback'] = client_feedback
            pdr_report['approval_status'] = client_feedback.get('approved', False)
        else:
            pdr_report['approval_status'] = 'Pending'
        
        # Create visualizations
        if self.pugh_matrix and PLOTTING_AVAILABLE:
            try:
                self.pugh_matrix.create_visualization('pdr_results.png')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not create visualization: {e}")
        
        # Export results
        if self.pugh_matrix:
            try:
                self.pugh_matrix.export_to_csv('pdr_results.csv')
                self.pugh_matrix.save_report('pdr_report.json')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not export results: {e}")
        
        # Generate DAR PDF
        if generate_dar:
            try:
                generate_dar('pdr_results.csv', 'pdr_dar_report.pdf')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not generate DAR PDF: {e}")
        
        self.results['final_approval'] = pdr_report
        
        if self.logger:
            self.logger.info("Final approval completed. PDR report generated.")
        return pdr_report
    
    def generate_pdr_report(self) -> Dict[str, Any]:
        """Generate comprehensive PDR report."""
        report = {
            'pdr_metadata': {
                'project_name': 'PDR Workflow Demo',
                'analysis_date': datetime.now().isoformat(),
                'workflow_version': '1.0',
                'total_steps_completed': len(self.results)
            },
            'executive_summary': {
                'total_requirements_analyzed': self.results.get('insights', {}).get('total_requirements', 0),
                'alternatives_evaluated': self.results.get('ideation', {}).get('alternatives_count', 0),
                'best_alternative': self.results.get('brainstorming', {}).get('best_alternative', 'N/A'),
                'final_recommendation': self.results.get('concept_selection', {}).get('final_recommendation', {})
            },
            'detailed_analysis': self.results,
            'recommendations': self._generate_final_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        return report
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations based on analysis."""
        recommendations = []
        
        if 'brainstorming' in self.results:
            report = self.results['brainstorming']['detailed_report']
            best_alt = report['summary']['best_alternative']
            best_score = report['summary']['best_score']
            
            recommendations.append(f"Proceed with alternative {best_alt} (Score: {best_score})")
            
            # Add specific recommendations based on criteria performance
            criteria_summary = report['criteria_summary']
            for criteria_id, summary in criteria_summary.items():
                if summary['avg_rating'] < 0:
                    recommendations.append(f"Focus on improving {criteria_id} in final design")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for the project."""
        return [
            "Detailed Design Phase",
            "Prototype Development",
            "Testing and Validation",
            "Manufacturing Planning",
            "Cost Optimization"
        ]
    
    def run_complete_workflow(self, document_paths: List[str], 
                             alternatives_data: List[Dict[str, Any]],
                             ratings_data: Dict[str, Dict[str, int]],
                             client_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the complete PDR workflow."""
        if self.logger:
            self.logger.info("Starting complete PDR workflow")
        
        try:
            # Step 1: Document Analysis
            step1_results = self.step_1_document_analysis(document_paths)
            
            # Step 2: Requirements Extraction
            step2_results = self.step_2_requirements_extraction()
            
            # Step 3: Ideation
            step3_results = self.step_3_ideation(alternatives_data)
            
            # Step 4: Brainstorming
            step4_results = self.step_4_brainstorming(ratings_data)
            
            # Step 5: Concept Selection
            step5_results = self.step_5_concept_selection()
            
            # Step 6: Final Approval
            step6_results = self.step_6_final_approval(client_feedback)
            
            if self.logger:
                self.logger.info("Complete PDR workflow finished successfully")
            return self.results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in PDR workflow: {e}")
            else:
                print(f"Error in PDR workflow: {e}")
            raise

def main():
    """Example usage of PDR Workflow."""
    # Initialize PDR workflow
    pdr = PDRWorkflow()
    
    # Sample document paths (replace with actual documents)
    document_paths = [
        'data/sample_document1.pdf',
        'data/sample_document2.txt'
    ]
    
    # Sample alternatives data
    alternatives_data = [
        {
            'id': 'ALT_001',
            'description': 'Aluminum Enclosure',
            'specifications': {
                'material': 'Aluminum',
                'cost': 150,
                'weight': 2.5,
                'corrosion_resistance': 'High',
                'manufacturing_complexity': 'Medium'
            }
        },
        {
            'id': 'ALT_002',
            'description': 'Stainless Steel Enclosure',
            'specifications': {
                'material': 'Stainless Steel',
                'cost': 300,
                'weight': 4.0,
                'corrosion_resistance': 'Very High',
                'manufacturing_complexity': 'High'
            }
        },
        {
            'id': 'ALT_003',
            'description': 'Plastic Enclosure',
            'specifications': {
                'material': 'ABS Plastic',
                'cost': 80,
                'weight': 1.5,
                'corrosion_resistance': 'Medium',
                'manufacturing_complexity': 'Low'
            }
        }
    ]
    
    # Sample ratings from cross-functional team
    ratings_data = {
        'ALT_002': {'R1': 1, 'R2': 1, 'R3': -1},  # Better compliance, better technical, higher cost
        'ALT_003': {'R1': -1, 'R2': -1, 'R3': 1}   # Worse compliance, worse technical, lower cost
    }
    
    # Sample client feedback
    client_feedback = {
        'approved': True,
        'comments': 'Recommendation accepted. Proceed with detailed design.',
        'additional_requirements': ['Add thermal management', 'Include EMI shielding']
    }
    
    # Run complete workflow
    try:
        results = pdr.run_complete_workflow(
            document_paths=document_paths,
            alternatives_data=alternatives_data,
            ratings_data=ratings_data,
            client_feedback=client_feedback
        )
        
        print("PDR Workflow completed successfully!")
        print(f"Best alternative: {results['brainstorming']['best_alternative']}")
        print(f"Final recommendation: {results['concept_selection']['final_recommendation']['alternative_id']}")
        
    except Exception as e:
        print(f"Error in PDR workflow: {e}")

if __name__ == "__main__":
    main() 
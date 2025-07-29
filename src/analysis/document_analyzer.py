import pandas as pd
import numpy as np
import re
import yaml
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import logging

# Optional imports for PDF processing
try:
    import camelot
    import tabula
    import pdfplumber
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    print("Warning: camelot/tabula/pdfplumber not available. PDF processing disabled.")

class DocumentAnalyzer:
    """
    Analyzes documents to extract T&C, compliance standards, and key requirements
    for PDR (Preliminary Design Review) decision making.
    """
    
    def __init__(self):
        self.compliance_keywords = [
            'iso', 'astm', 'asme', 'ieee', 'ul', 'ce', 'fcc', 'rohs', 'reach',
            'compliance', 'standard', 'certification', 'requirement', 'specification',
            'regulation', 'guideline', 'protocol', 'framework', 'approval', 'certified',
            'qualified', 'tested', 'verified', 'validated', 'approved', 'accepted'
        ]
        
        self.technical_keywords = [
            'material', 'component', 'specification', 'dimension', 'tolerance',
            'performance', 'reliability', 'durability', 'safety', 'quality',
            'testing', 'validation', 'verification', 'manufacturing', 'assembly',
            'design', 'engineering', 'technical', 'mechanical', 'electrical',
            'structural', 'functional', 'operational', 'maintenance', 'service',
            'installation', 'configuration', 'setup', 'calibration', 'inspection'
        ]
        
        self.cost_keywords = [
            'cost', 'price', 'budget', 'economic', 'financial', 'expense',
            'investment', 'roi', 'value', 'efficiency', 'optimization',
            'dollar', 'currency', 'payment', 'invoice', 'quote', 'estimate',
            'pricing', 'rate', 'fee', 'charge', 'expense', 'overhead'
        ]
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('document_analysis.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
            self.logger = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods."""
        if not PDF_PROCESSING_AVAILABLE:
            print("Warning: PDF processing not available")
            return ""
        
        text = ""
        
        try:
            # Try pdfplumber first (best for text extraction)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    print(f"Successfully extracted text using pdfplumber")
                    return text
            except Exception as e:
                print(f"pdfplumber failed: {e}")
            
            # Try camelot for tables
            try:
                tables = camelot.read_pdf(pdf_path, pages='all')
                for table in tables:
                    text += table.df.to_string() + "\n"
                print(f"Successfully extracted tables using camelot")
            except Exception as e:
                print(f"camelot failed: {e}")
            
            # Try tabula as fallback
            if not text.strip():
                try:
                    tables = tabula.read_pdf(pdf_path, pages='all')
                    for table in tables:
                        text += table.to_string() + "\n"
                    print(f"Successfully extracted tables using tabula")
                except Exception as e:
                    print(f"tabula failed: {e}")
            
            return text
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting text from PDF: {e}")
            else:
                print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_requirements(self, text: str) -> Dict[str, List[str]]:
        """Extract requirements from text."""
        requirements = {
            'compliance': [],
            'technical': [],
            'cost': [],
            'general': []
        }
        
        if not text:
            print("No text to analyze")
            return requirements
        
        print(f"Analyzing text of length: {len(text)} characters")
        
        lines = text.split('\n')
        print(f"Found {len(lines)} lines of text")
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            line_lower = line.lower()
            
            # Extract compliance requirements (more specific matching)
            compliance_matches = [kw for kw in self.compliance_keywords if kw in line_lower]
            if compliance_matches and any(kw in line_lower for kw in ['iso', 'astm', 'asme', 'ieee', 'ul', 'ce', 'fcc', 'rohs', 'reach', 'certification', 'standard']):
                requirements['compliance'].append(line.strip())
                print(f"Found compliance requirement: {line.strip()[:50]}...")
            
            # Extract cost-related requirements (check before technical to avoid conflicts)
            elif any(keyword in line_lower for keyword in self.cost_keywords):
                requirements['cost'].append(line.strip())
                print(f"Found cost requirement: {line.strip()[:50]}...")
            
            # Extract technical requirements
            elif any(keyword in line_lower for keyword in self.technical_keywords):
                requirements['technical'].append(line.strip())
                print(f"Found technical requirement: {line.strip()[:50]}...")
            
            # General requirements (lines with numbers or bullet points)
            elif re.search(r'^\d+\.|^[•\-*]|^[A-Z]\.', line.strip()):
                requirements['general'].append(line.strip())
                print(f"Found general requirement: {line.strip()[:50]}...")
        
        print(f"Extracted requirements: {sum(len(reqs) for reqs in requirements.values())} total")
        return requirements
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze a document and extract key information."""
        print(f"Starting analysis of: {file_path}")
        
        if self.logger:
            self.logger.info(f"Analyzing document: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return {}
        
        # Extract text based on file type
        text = ""
        try:
            if file_path.lower().endswith('.pdf'):
                print(f"Extracting text from PDF: {file_path}")
                text = self.extract_text_from_pdf(file_path)
                print(f"Extracted {len(text)} characters from PDF")
            elif file_path.lower().endswith('.txt'):
                print(f"Reading text file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Read {len(text)} characters from text file")
            else:
                error_msg = f"Unsupported file type: {file_path}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(error_msg)
                return {}
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return {}
        
        # Extract requirements
        print("Extracting requirements from text...")
        requirements = self.extract_requirements(text)
        
        # Generate summary
        print("Generating summary...")
        summary = self.generate_summary(requirements)
        
        result = {
            'file_path': file_path,
            'text_length': len(text),
            'requirements': requirements,
            'summary': summary,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"Analysis complete. Found {sum(len(reqs) for reqs in requirements.values())} requirements")
        return result
    
    def generate_summary(self, requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate a summary of requirements for SME and architect decision making."""
        summary = {
            'total_requirements': sum(len(reqs) for reqs in requirements.values()),
            'compliance_count': len(requirements['compliance']),
            'technical_count': len(requirements['technical']),
            'cost_count': len(requirements['cost']),
            'general_count': len(requirements['general']),
            'key_insights': [],
            'recommendations': []
        }
        
        # Generate key insights
        if requirements['compliance']:
            summary['key_insights'].append(f"Found {len(requirements['compliance'])} compliance requirements")
        
        if requirements['technical']:
            summary['key_insights'].append(f"Identified {len(requirements['technical'])} technical specifications")
        
        if requirements['cost']:
            summary['key_insights'].append(f"Extracted {len(requirements['cost'])} cost-related considerations")
        
        # Generate recommendations
        if summary['compliance_count'] > 0:
            summary['recommendations'].append("Prioritize compliance requirements in design decisions")
        
        if summary['technical_count'] > 0:
            summary['recommendations'].append("Consider technical specifications for material selection")
        
        if summary['cost_count'] > 0:
            summary['recommendations'].append("Include cost analysis in Pugh matrix criteria")
        
        return summary
    
    def analyze_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple documents and create a comprehensive report."""
        all_analyses = []
        
        for file_path in file_paths:
            analysis = self.analyze_document(file_path)
            if analysis:
                all_analyses.append(analysis)
        
        # Combine all requirements
        combined_requirements = {
            'compliance': [],
            'technical': [],
            'cost': [],
            'general': []
        }
        
        for analysis in all_analyses:
            for category, reqs in analysis['requirements'].items():
                combined_requirements[category].extend(reqs)
        
        # Remove duplicates
        for category in combined_requirements:
            combined_requirements[category] = list(set(combined_requirements[category]))
        
        # Generate overall summary
        overall_summary = self.generate_summary(combined_requirements)
        overall_summary['documents_analyzed'] = len(all_analyses)
        overall_summary['file_paths'] = [analysis['file_path'] for analysis in all_analyses]
        
        return {
            'individual_analyses': all_analyses,
            'combined_requirements': combined_requirements,
            'overall_summary': overall_summary
        }

def main():
    """Main function for document analysis."""
    analyzer = DocumentAnalyzer()
    
    # Example usage
    sample_docs = [
        'data/sample_document1.pdf',
        'data/sample_document2.txt'
    ]
    
    # Check which files exist
    existing_files = [f for f in sample_docs if Path(f).exists()]
    
    if existing_files:
        results = analyzer.analyze_multiple_documents(existing_files)
        
        # Save results
        try:
            with open('document_analysis_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print("Document analysis completed. Results saved to document_analysis_results.json")
        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("No sample documents found. Please add documents to the data/ directory.")

if __name__ == "__main__":
    main() 
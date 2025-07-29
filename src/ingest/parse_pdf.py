import pandas as pd
import camelot
import tabula
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

def parse_pdf_with_camelot(pdf_path: str, output_csv: str) -> pd.DataFrame:
    """
    Parse PDF tables using camelot-py library.
    
    Args:
        pdf_path: Path to the PDF file
        output_csv: Path to save the extracted CSV data
        
    Returns:
        DataFrame containing extracted table data
    """
    try:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file {pdf_path} not found")
            return pd.DataFrame()
        
        # Extract tables from PDF
        tables = camelot.read_pdf(pdf_path, pages='all')
        
        if not tables:
            print("No tables found in PDF")
            return pd.DataFrame()
        
        # Combine all tables
        all_data = []
        for i, table in enumerate(tables):
            df = table.df
            df['table_index'] = i
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            combined_df.to_csv(output_csv, index=False)
            print(f"Extracted {len(combined_df)} rows from {len(tables)} tables")
            print(f"Data saved to {output_csv}")
            
            return combined_df
        else:
            print("No data extracted from PDF")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error parsing PDF with camelot: {e}")
        return pd.DataFrame()

def parse_pdf_with_tabula(pdf_path: str, output_csv: str) -> pd.DataFrame:
    """
    Parse PDF tables using tabula-py library.
    
    Args:
        pdf_path: Path to the PDF file
        output_csv: Path to save the extracted CSV data
        
    Returns:
        DataFrame containing extracted table data
    """
    try:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file {pdf_path} not found")
            return pd.DataFrame()
        
        # Extract tables from PDF
        tables = tabula.read_pdf(pdf_path, pages='all')
        
        if not tables:
            print("No tables found in PDF")
            return pd.DataFrame()
        
        # Combine all tables
        all_data = []
        for i, table in enumerate(tables):
            table['table_index'] = i
            all_data.append(table)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            combined_df.to_csv(output_csv, index=False)
            print(f"Extracted {len(combined_df)} rows from {len(tables)} tables")
            print(f"Data saved to {output_csv}")
            
            return combined_df
        else:
            print("No data extracted from PDF")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error parsing PDF with tabula: {e}")
        return pd.DataFrame()

def parse_pdf(pdf_path: str, output_csv: str, method: str = 'camelot') -> pd.DataFrame:
    """
    Parse PDF file and extract table data.
    
    Args:
        pdf_path: Path to the PDF file
        output_csv: Path to save the extracted CSV data
        method: Method to use ('camelot' or 'tabula')
        
    Returns:
        DataFrame containing extracted table data
    """
    try:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file {pdf_path} not found")
            return pd.DataFrame()
        
        print(f"Parsing PDF: {pdf_path}")
        print(f"Using method: {method}")
        
        if method.lower() == 'camelot':
            return parse_pdf_with_camelot(pdf_path, output_csv)
        elif method.lower() == 'tabula':
            return parse_pdf_with_tabula(pdf_path, output_csv)
        else:
            print(f"Unknown method: {method}. Using camelot as default.")
            return parse_pdf_with_camelot(pdf_path, output_csv)
            
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return pd.DataFrame()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file {pdf_path} not found")
            return ""
        
        # Try camelot first for text extraction
        tables = camelot.read_pdf(pdf_path, pages='all')
        text_content = ""
        
        for table in tables:
            # Convert table to text
            for row in table.df.values:
                text_content += " ".join(str(cell) for cell in row) + "\n"
        
        if not text_content.strip():
            # Fallback to tabula
            tables = tabula.read_pdf(pdf_path, pages='all')
            for table in tables:
                text_content += table.to_string() + "\n"
        
        return text_content.strip()
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def analyze_pdf_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF structure and extract metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF analysis results
    """
    try:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file {pdf_path} not found")
            return {}
        
        analysis = {
            'file_path': pdf_path,
            'file_size': Path(pdf_path).stat().st_size,
            'tables_found': 0,
            'text_content': "",
            'extraction_method': None
        }
        
        # Try to extract tables
        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            analysis['tables_found'] = len(tables)
            analysis['extraction_method'] = 'camelot'
        except:
            try:
                tables = tabula.read_pdf(pdf_path, pages='all')
                analysis['tables_found'] = len(tables)
                analysis['extraction_method'] = 'tabula'
            except:
                analysis['extraction_method'] = 'none'
        
        # Extract text content
        analysis['text_content'] = extract_text_from_pdf(pdf_path)
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing PDF structure: {e}")
        return {}

def main():
    """Example usage of PDF parsing functions."""
    # Example PDF file path
    pdf_file = "data/sample_document.pdf"
    
    if Path(pdf_file).exists():
        # Analyze PDF structure
        analysis = analyze_pdf_structure(pdf_file)
        print("PDF Analysis Results:")
        for key, value in analysis.items():
            if key != 'text_content':  # Skip long text content
                print(f"{key}: {value}")
        
        # Parse PDF with different methods
        output_csv = "extracted_data.csv"
        
        # Try camelot
        df_camelot = parse_pdf(pdf_file, "camelot_output.csv", "camelot")
        if not df_camelot.empty:
            print(f"Camelot extracted {len(df_camelot)} rows")
        
        # Try tabula
        df_tabula = parse_pdf(pdf_file, "tabula_output.csv", "tabula")
        if not df_tabula.empty:
            print(f"Tabula extracted {len(df_tabula)} rows")
        
    else:
        print(f"Sample PDF file {pdf_file} not found")

if __name__ == "__main__":
    main()
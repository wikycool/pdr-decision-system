#!/usr/bin/env python3
"""
Test PDF report generation and show file locations
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from reports.dar_report import generate_dar
    print("✅ Successfully imported DAR report generator")
except ImportError as e:
    print(f"❌ Error importing DAR report generator: {e}")
    sys.exit(1)

def test_pdf_generation():
    """Test PDF generation and show file locations."""
    print("🚀 Testing PDF report generation...")
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"📁 Current working directory: {current_dir}")
    
    # Check if CSV file exists
    csv_file = "pdr_results.csv"
    if os.path.exists(csv_file):
        print(f"✅ Found CSV file: {csv_file}")
        print(f"   Size: {os.path.getsize(csv_file)} bytes")
    else:
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    # Generate PDF report
    pdf_file = "pdr_dar_report.pdf"
    print(f"\n📄 Generating PDF report: {pdf_file}")
    
    try:
        generate_dar(csv_file, pdf_file)
        
        # Check if PDF was created
        if os.path.exists(pdf_file):
            print(f"✅ PDF report generated successfully!")
            print(f"   File: {pdf_file}")
            print(f"   Location: {os.path.abspath(pdf_file)}")
            print(f"   Size: {os.path.getsize(pdf_file)} bytes")
            
            # Show file location details
            print(f"\n📋 File Details:")
            print(f"   Full Path: {os.path.abspath(pdf_file)}")
            print(f"   Directory: {os.path.dirname(os.path.abspath(pdf_file))}")
            print(f"   Filename: {os.path.basename(pdf_file)}")
            
        else:
            print(f"❌ PDF file was not created: {pdf_file}")
            
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

def show_all_output_files():
    """Show all output files in the project directory."""
    print("\n📁 All Output Files in Project Directory:")
    
    current_dir = os.getcwd()
    output_files = [
        "pdr_results.csv",
        "pdr_report.json", 
        "pdr_results.png",
        "pdr_dar_report.pdf"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} (not found)")

def main():
    """Main function to test PDF generation."""
    print("🔍 PDF Report Generation Test")
    print("=" * 50)
    
    # Test PDF generation
    test_pdf_generation()
    
    # Show all output files
    show_all_output_files()
    
    print("\n💡 How to open the PDF report:")
    print("   1. Navigate to your project directory")
    print("   2. Look for 'pdr_dar_report.pdf'")
    print("   3. Double-click to open with default PDF viewer")
    print("   4. Or right-click → Open with → Choose PDF viewer")

if __name__ == "__main__":
    main() 
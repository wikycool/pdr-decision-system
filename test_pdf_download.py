#!/usr/bin/env python3
"""
Test PDF download functionality
"""

import os
import sys
import base64
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from reports.dar_report import generate_dar
    print("✅ Successfully imported DAR report generator")
except ImportError as e:
    print(f"❌ Error importing DAR report generator: {e}")
    sys.exit(1)

def create_download_link(val, filename):
    """Create a download link for files."""
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def test_pdf_download():
    """Test PDF download functionality."""
    print("🚀 Testing PDF download functionality...")
    
    # Create sample CSV data
    import pandas as pd
    sample_data = {
        'alternative_id': ['ALT_001', 'ALT_002', 'ALT_003'],
        'description': ['Aluminum Enclosure', 'Stainless Steel Enclosure', 'Plastic Enclosure'],
        'final_score': [15.5, 12.3, 8.7],
        'total_weight': [10.0, 10.0, 10.0]
    }
    
    df = pd.DataFrame(sample_data)
    temp_csv = 'temp_test_data.csv'
    df.to_csv(temp_csv, index=False)
    
    print(f"✅ Created sample CSV data: {temp_csv}")
    
    # Generate PDF
    pdf_filename = 'test_analysis_report.pdf'
    try:
        generate_dar(temp_csv, pdf_filename)
        print(f"✅ Generated PDF: {pdf_filename}")
        
        # Read the generated PDF
        with open(pdf_filename, 'rb') as f:
            pdf_data = f.read()
        
        print(f"✅ Read PDF data: {len(pdf_data)} bytes")
        
        # Create download link
        download_link = create_download_link(pdf_data, pdf_filename)
        print("✅ Created download link")
        
        # Save the HTML link to a file for testing
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDF Download Test</title>
</head>
<body>
    <h1>PDF Download Test</h1>
    <p>Click the link below to download the PDF report:</p>
    {download_link}
</body>
</html>
"""
        
        with open('test_download.html', 'w') as f:
            f.write(html_content)
        
        print("✅ Created test HTML file: test_download.html")
        print("📁 Files created:")
        print(f"   - {temp_csv} (sample data)")
        print(f"   - {pdf_filename} (PDF report)")
        print(f"   - test_download.html (download test)")
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error in PDF generation: {e}")

def main():
    """Main test function."""
    print("🔍 PDF Download Functionality Test")
    print("=" * 50)
    
    test_pdf_download()
    
    print("\n💡 How to test in Streamlit:")
    print("   1. Run: python run_app.py")
    print("   2. Go to: http://localhost:8501")
    print("   3. Navigate to '📊 Results' or '🚀 One-Click PDR Analysis'")
    print("   4. Click '📄 Download PDF Report' or '📄 Generate Analysis Report'")
    print("   5. The PDF should download automatically in your browser")

if __name__ == "__main__":
    main() 
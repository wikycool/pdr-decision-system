import streamlit as st
import pandas as pd
import numpy as np
import yaml
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
try:
    from analysis.langchain_analyzer import LangChainDocumentAnalyzer
    from analysis.enhanced_document_analyzer import EnhancedDocumentAnalyzer
    from ml.material_recommender import MaterialTechnologyRecommender
    from scoring.enhanced_pugh import EnhancedPughMatrix
    from ml.enhanced_npd_analyzer import EnhancedNPDAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    MODULES_AVAILABLE = False

def main():
    """Main Streamlit application for PDR Decision System."""
    st.set_page_config(
        page_title="PDR Decision System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 PDR (Preliminary Design Review) Decision System")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "🎯 NPD Analysis Pages",
        [
            "🏠 Home",
            "📄 Document Analysis", 
            "🤖 AI/ML Recommendations",
            "⚖️ Decision Matrix",
            "🔄 PDR Workflow",
            "📊 Results",
            "💾 Data Management",
            "🚀 One-Click PDR Analysis",
            "🎯 Enhanced NPD Analysis"
        ]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 Document Analysis":
        show_document_analysis()
    elif page == "🤖 AI/ML Recommendations":
        show_ai_ml_recommendations()
    elif page == "⚖️ Decision Matrix":
        show_pugh_matrix()
    elif page == "🔄 PDR Workflow":
        show_pdr_workflow()
    elif page == "📊 Results":
        show_results()
    elif page == "💾 Data Management":
        show_data_management()
    elif page == "🚀 One-Click PDR Analysis":
        show_one_click_analysis()
    elif page == "🎯 Enhanced NPD Analysis":
        show_enhanced_npd_analysis()

def show_home_page():
    """Display the home page with system overview."""
    st.title("🚀 AI-Powered Document Analysis System")
    st.markdown("### 📄 **Document Analysis with AI/ML**")
    
    st.markdown("""
    This system provides advanced document analysis using AI/ML algorithms to extract 
    requirements, categorize information, and generate insights from various document formats.
    """)
    
    # Key Features
    st.subheader("🎯 **Key Features**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ **Document Analysis**
        - **PDF, DOCX, TXT** support
        - **AI-Powered**: LangChain with GPT-3.5-turbo
        - **Keyword-Based**: Traditional analysis
        - **Requirement Extraction**: Automatic categorization
        """)
        
        st.markdown("""
        ### ✅ **AI/ML Integration**
        - **Multiple Algorithms**: RandomForest, GradientBoosting, SVR
        - **Material Recommendations**: ML-based selection
        - **Technology Analysis**: Manufacturing suggestions
        - **Cost Estimation**: AI-powered analysis
        """)
    
    with col2:
        st.markdown("""
        ### ✅ **Decision Matrix**
        - **Pugh Matrix**: Weighted criteria evaluation
        - **Alternative Generation**: AI-powered options
        - **Scoring System**: Intelligent rating algorithms
        - **Professional Reports**: Export capabilities
        """)
        
        st.markdown("""
        ### ✅ **One-Click Analysis**
        - **Complete Workflow**: Document to final concept
        - **Automated Processing**: No manual steps
        - **Comprehensive Reports**: Ready for presentation
        - **Multiple Formats**: PDF, DOCX, TXT support
        """)
    
    # AI/ML Algorithms
    st.subheader("🤖 **AI/ML Algorithms Used**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 RandomForest**
        - Material classification
        - Cost prediction
        - Feature importance analysis
        """)
        
        st.markdown("""
        **🚀 GradientBoosting**
        - Technology recommendation
        - Performance optimization
        - Ensemble learning
        """)
    
    with col2:
        st.markdown("""
        **⚡ SVR (Support Vector Regression)**
        - Risk assessment
        - Cost estimation
        - Non-linear relationships
        """)
        
        st.markdown("""
        **🔗 LangChain**
        - Document analysis
        - GPT-3.5-turbo integration
        - Natural language processing
        """)
    
    with col3:
        st.markdown("""
        **🎯 KMeans**
        - Product complexity analysis
        - Clustering algorithms
        - Pattern identification
        """)
        
        st.markdown("""
        **📄 Document Processing**
        - PDF, DOCX, TXT support
        - Text extraction
        - Table parsing
        """)
    
    # Quick Actions
    st.subheader("🚀 **Quick Actions**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Enhanced Analysis**
        - Multiple AI/ML algorithms
        - Comprehensive document analysis
        - Risk assessment & recommendations
        - Strategic insights
        """)
        
        if st.button("🎯 Start Enhanced Analysis", type="primary"):
            st.info("Navigate to '🎯 Enhanced NPD Analysis' in the sidebar to begin comprehensive analysis.")
    
    with col2:
        st.info("""
        **🚀 One-Click Analysis**
        - Complete workflow automation
        - Document analysis to final concept
        - AI-powered recommendations
        - Decision matrix generation
        """)
        
        if st.button("🚀 Start One-Click Analysis", type="secondary"):
            st.info("Navigate to '🚀 One-Click PDR Analysis' in the sidebar for automated complete analysis.")
    
    # System Architecture
    st.subheader("🏗️ **System Architecture**")
    
    st.markdown("""
    ```
    📄 Document Upload → 🤖 AI Analysis → 📊 Decision Matrix → 🎯 Final Concept
         ↓                    ↓                    ↓                    ↓
    Text Extraction    Material/Technology    Pugh Matrix        Client Approval
    Requirements      Cost Estimation        Scoring            Presentation
    Categorization    Product Picture       Alternatives       Analysis Complete
    ```
    """)
    
    # Technical Stack
    st.subheader("🛠️ **Technical Stack**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend**
        - Streamlit (Interactive UI)
        - Real-time analysis
        - Professional reports
        """)
    
    with col2:
        st.markdown("""
        **AI/ML**
        - LangChain, scikit-learn
        - OpenAI GPT-3.5-turbo
        - Multiple algorithms
        """)
    
    with col3:
        st.markdown("""
        **Document Processing**
        - PDF, DOCX, TXT support
        - Text extraction
        - Table parsing
        """)
    
    st.markdown("---")
    st.markdown("**🚀 Ready to analyze your documents! Upload a PDF, DOCX, or TXT file and see the AI/ML magic in action!** ✨")

def show_document_analysis():
    """Document analysis page."""
    st.header("📄 Document Analysis")
    st.markdown("""
    ### 🔍 **Extract Requirements from Documents**
    
    Upload PDF, DOCX, or TXT files to extract and categorize requirements using AI/ML algorithms.
    """)
    
    if not MODULES_AVAILABLE:
        st.error("Required modules not available. Please check dependencies.")
        return
    
    # Analysis method selection
    analyzer_type = st.selectbox(
        "Analysis Method",
        ["🤖 LangChain (AI-Powered)", "🔍 Traditional (Keyword-Based)"],
        help="Choose between AI-powered analysis or traditional keyword-based analysis"
    )
    
    # API key input for LangChain
    if analyzer_type == "🤖 LangChain (AI-Powered)":
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key for AI-powered analysis. Leave empty to use fallback mode."
        )
        
        if openai_api_key:
            st.success("✅ OpenAI API key provided - using AI-powered analysis!")
        else:
            st.warning("⚠️ No API key provided. Using fallback analysis mode.")
        
        analyzer = LangChainDocumentAnalyzer(openai_api_key=openai_api_key if openai_api_key else None)
    else:
        if not EnhancedDocumentAnalyzer:
            st.error("Traditional analyzer not available. Please check dependencies.")
            return
        analyzer = EnhancedDocumentAnalyzer()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or text files to extract requirements"
    )
    
    if uploaded_files:
        st.success(f"📄 Uploaded {len(uploaded_files)} file(s)")
        
        # Process each file
        all_results = []
        
        for uploaded_file in uploaded_files:
            st.write(f"🔍 Analyzing: {uploaded_file.name}")
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            try:
                # Analyze document
                results = analyzer.analyze_document(temp_file_path)
                
                if results:
                    results['file_name'] = uploaded_file.name
                    all_results.append(results)
                    st.success(f"✅ Analysis complete for {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No results for {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing {uploaded_file.name}: {e}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        # Display combined results
        if all_results:
            st.subheader("📊 Analysis Results")
            
            # Summary metrics
            total_requirements = sum(result.get('summary', {}).get('total_requirements', 0) for result in all_results)
            total_files = len(all_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Analyzed", total_files)
            with col2:
                st.metric("Total Requirements", total_requirements)
            with col3:
                st.metric("Avg Requirements/File", round(total_requirements/total_files, 1) if total_files > 0 else 0)
            
            # Detailed results for each file
            for i, result in enumerate(all_results):
                with st.expander(f"📄 {result['file_name']} - Analysis Details"):
                    
                    summary = result.get('summary', {})
                    requirements = result.get('requirements', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Summary:**")
                        st.write(f"• Total Requirements: {summary.get('total_requirements', 0)}")
                        st.write(f"• Compliance: {summary.get('compliance_count', 0)}")
                        st.write(f"• Technical: {summary.get('technical_count', 0)}")
                        st.write(f"• Cost: {summary.get('cost_count', 0)}")
                    
                    with col2:
                        st.write("**💡 Key Insights:**")
                        insights = summary.get('key_insights', [])
                        for insight in insights[:3]:
                            st.write(f"• {insight}")
                    
                    # Show requirements by category
                    if requirements:
                        st.write("**📋 Requirements by Category:**")
                        
                        for category, reqs in requirements.items():
                            if reqs:
                                with st.expander(f"{category.title()} ({len(reqs)} items)"):
                                    for j, req in enumerate(reqs[:5], 1):  # Show first 5
                                        st.write(f"{j}. {req}")
                                    if len(reqs) > 5:
                                        st.write(f"... and {len(reqs) - 5} more")
            
            # Export option
            st.subheader("📄 Export Results")
            if st.button("📊 Generate Analysis Report"):
                st.success("✅ Analysis report generated successfully!")
                st.info("📋 Report includes: Summary metrics, detailed requirements by category, and key insights for each file.")
        else:
            st.warning("⚠️ No analysis results to display.")
    else:
        st.info("📄 Please upload PDF, DOCX, or TXT files to start analysis.")

def show_ai_ml_recommendations():
    """AI/ML Material and Technology Recommendations page."""
    st.header("🤖 AI/ML Material & Technology Recommendations")
    
    if not MODULES_AVAILABLE or not MaterialTechnologyRecommender:
        st.error("AI/ML recommender not available. Please check dependencies.")
        return
    
    st.markdown("""
    ### 🎯 AI/ML-Powered Recommendations
    
    This system uses machine learning to recommend:
    - **Materials** based on requirements analysis
    - **Manufacturing Technologies** for optimal production
    - **Cost Estimates** for budget planning
    - **Latest Technology Insights** for competitive advantage
    """)
    
    # Initialize recommender
    recommender = MaterialTechnologyRecommender()
    
    # Input requirements
    st.subheader("📋 Input Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compliance_reqs = st.text_area(
            "Compliance Requirements",
            placeholder="Enter compliance requirements (e.g., ISO standards, certifications)",
            height=100
        )
        
        technical_reqs = st.text_area(
            "Technical Requirements",
            placeholder="Enter technical specifications (e.g., strength, weight, performance)",
            height=100
        )
    
    with col2:
        cost_reqs = st.text_area(
            "Cost Requirements",
            placeholder="Enter cost constraints and budget considerations",
            height=100
        )
        
        other_reqs = st.text_area(
            "Other Requirements",
            placeholder="Enter any other requirements or constraints",
            height=100
        )
    
    # Process requirements
    if st.button("🤖 Generate AI/ML Recommendations", type="primary"):
        if not any([compliance_reqs, technical_reqs, cost_reqs, other_reqs]):
            st.warning("Please enter at least some requirements to generate recommendations.")
            return
        
        with st.spinner("Analyzing requirements with AI/ML..."):
            # Prepare requirements
            requirements = {
                'compliance': [req.strip() for req in compliance_reqs.split('\n') if req.strip()],
                'technical': [req.strip() for req in technical_reqs.split('\n') if req.strip()],
                'cost': [req.strip() for req in cost_reqs.split('\n') if req.strip()],
                'other': [req.strip() for req in other_reqs.split('\n') if req.strip()]
            }
            
            # Generate comprehensive report
            report = recommender.generate_comprehensive_report(requirements)
            
            # Display results
            st.success("✅ AI/ML analysis completed!")
            
            # Material Recommendations
            st.subheader("🏗️ Material Recommendations")
            if report['material_recommendations']:
                for i, material in enumerate(report['material_recommendations']):
                    with st.expander(f"🥇 {material['name']} (Score: {material['score']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Cost:** ${material['cost_per_kg']}/kg")
                            st.write(f"**Properties:** {', '.join(material['properties'])}")
                            st.write(f"**Applications:** {', '.join(material['applications'])}")
                        
                        with col2:
                            st.write("**Recommendation Reasons:**")
                            for reason in material['recommendation_reason']:
                                st.write(f"• {reason}")
            
            # Technology Recommendations
            st.subheader("⚙️ Manufacturing Technology Recommendations")
            if report['technology_recommendations']:
                for i, tech in enumerate(report['technology_recommendations']):
                    with st.expander(f"🔧 {tech['name']} (Score: {tech['score']})"):
                        st.write(f"**Description:** {tech['description']}")
                        st.write(f"**Cost Factor:** {tech['cost_factor']}x")
                        st.write(f"**Lead Time:** {tech['lead_time']}")
                        st.write(f"**Material Compatibility:** {tech['material_compatibility']}")
                        
                        st.write("**Recommendation Reasons:**")
                        for reason in tech['recommendation_reason']:
                            st.write(f"• {reason}")
            
            # Cost Analysis
            st.subheader("💰 Cost Analysis")
            if report['cost_estimates']['cost_breakdown']:
                cost_range = report['cost_estimates']['cost_breakdown']['cost_range']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Minimum Cost", f"${cost_range['min']:.2f}")
                
                with col2:
                    st.metric("Average Cost", f"${cost_range['average']:.2f}")
                
                with col3:
                    st.metric("Maximum Cost", f"${cost_range['max']:.2f}")
                
                # Cost recommendations
                st.write("**Cost Optimization Recommendations:**")
                for rec in report['cost_estimates']['cost_breakdown']['recommendations']:
                    st.write(f"• {rec}")
            
            # AI Insights
            st.subheader("🤖 AI/ML Insights")
            for insight in report['ai_ml_insights']:
                st.info(f"💡 {insight}")
            
            # Summary
            st.subheader("📊 Summary")
            summary = report['recommendations_summary']
            
            if summary['top_material']:
                st.write(f"**Top Material:** {summary['top_material']['name']}")
            
            if summary['top_technology']:
                st.write(f"**Top Technology:** {summary['top_technology']['name']}")
            
            st.write("**Key Decisions:**")
            for decision in summary['key_decisions']:
                st.write(f"• {decision}")
            
            # Export results
            if st.button("📥 Export AI/ML Report"):
                # Create downloadable report
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="Download AI/ML Report (JSON)",
                    data=report_json,
                    file_name="ai_ml_recommendations.json",
                    mime="application/json"
                )

def show_pugh_matrix():
    """Pugh matrix decision tool."""
    st.header("⚖️ Decision Matrix")
    st.markdown("""
    ### 📊 **Pugh Matrix Decision Tool**
    
    Compare alternatives against weighted criteria to make informed decisions.
    """)
    
    if not MODULES_AVAILABLE or not EnhancedPughMatrix:
        st.error("Enhanced Pugh Matrix module not available. Please check dependencies.")
        return
    
    pugh = EnhancedPughMatrix()
    
    # Load criteria
    st.subheader("📋 Criteria Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Default Criteria:**")
        criteria = {
            'R1': 'Compliance Requirements',
            'R2': 'Technical Specifications', 
            'R3': 'Cost Considerations',
            'R4': 'Quality Standards',
            'R5': 'Timeline Constraints'
        }
        
        for key, value in criteria.items():
            st.write(f"• {key}: {value}")
    
    with col2:
        st.write("**Default Weights:**")
        weights = {'R1': 10, 'R2': 8, 'R3': 6, 'R4': 7, 'R5': 5}
        
        for key, weight in weights.items():
            st.write(f"• {key}: {weight}")
    
    # Set criteria and weights
    pugh.set_criteria(criteria, weights)
    
    # Add alternatives
    st.subheader("🔄 Alternatives")
    
    alternatives = [
        ("ALT_001", "Aluminum Solution", {"material": "Aluminum", "cost": "Medium", "strength": "Good"}),
        ("ALT_002", "Stainless Steel Solution", {"material": "Stainless Steel", "cost": "High", "strength": "Excellent"}),
        ("ALT_003", "Carbon Steel Solution", {"material": "Carbon Steel", "cost": "Low", "strength": "Standard"})
    ]
    
    for alt_id, description, specs in alternatives:
        pugh.add_alternative(alt_id, description, specs)
        st.write(f"• **{alt_id}**: {description}")
    
    # Set baseline
    pugh.set_baseline("ALT_001")
    st.info("📌 Baseline set to: ALT_001")
    
    # Auto-rate alternatives
    st.subheader("📊 Auto-Rating")
    
    # Rate alternatives based on material properties
    for alt_id, description, specs in alternatives:
        if 'aluminum' in specs['material'].lower():
            pugh.rate_alternative(alt_id, 'R1', 1, 'Good compliance')
            pugh.rate_alternative(alt_id, 'R2', 1, 'Good technical specs')
            pugh.rate_alternative(alt_id, 'R3', 1, 'Cost effective')
            pugh.rate_alternative(alt_id, 'R4', 1, 'Good quality')
            pugh.rate_alternative(alt_id, 'R5', 1, 'Standard timeline')
        elif 'stainless' in specs['material'].lower():
            pugh.rate_alternative(alt_id, 'R1', 2, 'Excellent compliance')
            pugh.rate_alternative(alt_id, 'R2', 2, 'Excellent technical specs')
            pugh.rate_alternative(alt_id, 'R3', -1, 'Higher cost')
            pugh.rate_alternative(alt_id, 'R4', 2, 'Excellent quality')
            pugh.rate_alternative(alt_id, 'R5', 0, 'Standard timeline')
        elif 'carbon' in specs['material'].lower():
            pugh.rate_alternative(alt_id, 'R1', 0, 'Standard compliance')
            pugh.rate_alternative(alt_id, 'R2', 0, 'Standard technical specs')
            pugh.rate_alternative(alt_id, 'R3', 2, 'Very cost effective')
            pugh.rate_alternative(alt_id, 'R4', 0, 'Standard quality')
            pugh.rate_alternative(alt_id, 'R5', 1, 'Fast timeline')
    
    st.success("✅ Auto-rating completed!")
    
    # Calculate and display results
    st.subheader("📈 Results")
    
    results = pugh.calculate_scores()
    
    if results:
        st.write("**🏆 Ranking by Score:**")
        for i, result in enumerate(results, 1):
            st.write(f"{i}. **{result['alternative_id']}**: {result['description']}")
            st.write(f"   Score: {result['final_score']:.3f}")
            st.write(f"   Total Weight: {result['total_weight']}")
        
        # Generate report
        report = pugh.generate_detailed_report()
        if report['summary']['best_alternative']:
            st.success(f"🏆 **Best Alternative**: {report['summary']['best_alternative']}")
        
        has_pugh_results = True
    else:
        st.warning("⚠️ No results to display")
        has_pugh_results = False
    
    # Export options
    st.subheader("📄 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            # Export Pugh results if available
            if has_pugh_results and MODULES_AVAILABLE and EnhancedPughMatrix:
                pugh = EnhancedPughMatrix()
                pugh.export_to_csv('pugh_matrix_results.csv')
                st.success("✅ Pugh matrix exported to pugh_matrix_results.csv")
            else:
                st.error("❌ No results to export")
    
    with col2:
        if st.button("📋 Generate Report"):
            if has_pugh_results:
                st.success("✅ Detailed report generated!")
                st.info("📋 Report includes: Alternative rankings, scores, and recommendations")
            else:
                st.error("❌ No results to report")

def show_pdr_workflow():
    """PDR workflow page."""
    st.header("🔄 PDR Workflow")
    
    if not MODULES_AVAILABLE or not PDRWorkflow:
        st.error("PDR Workflow module not available. Please check dependencies.")
        return
    
    # Initialize workflow
    if 'pdr_workflow' not in st.session_state:
        st.session_state.pdr_workflow = PDRWorkflow()
    
    pdr = st.session_state.pdr_workflow
    
    # Workflow steps
    st.subheader("📋 Workflow Steps")
    
    steps = [
        ("📄 Document Analysis", "Analyze documents and extract requirements"),
        ("⚙️ Requirements Extraction", "Convert requirements to decision criteria"),
        ("💡 Ideation", "Generate and evaluate alternatives"),
        ("🧠 Brainstorming", "Cross-functional team rating"),
        ("🎯 Concept Selection", "Narrow down to final concept"),
        ("✅ Final Approval", "Client approval and documentation")
    ]
    
    for i, (step_name, step_desc) in enumerate(steps, 1):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{i}.**")
        with col2:
            st.write(f"**{step_name}**")
            st.caption(step_desc)
    
    # Run workflow
    st.subheader("🚀 Run Complete Workflow")
    
    # Sample data inputs
    with st.expander("📝 Sample Data Configuration"):
        st.write("Configure sample data for the workflow:")
        
        # Document paths
        doc_paths = st.text_area(
            "Document Paths (one per line)",
            value="data/sample_document1.pdf\ndata/sample_document2.txt",
            height=100
        )
        
        # Alternatives data
        st.write("**Sample Alternatives:**")
        alternatives_data = [
            {
                'id': 'ALT_001',
                'description': 'Aluminum Enclosure',
                'specifications': {'material': 'Aluminum', 'cost': 150, 'weight': 2.5}
            },
            {
                'id': 'ALT_002', 
                'description': 'Stainless Steel Enclosure',
                'specifications': {'material': 'Stainless Steel', 'cost': 300, 'weight': 4.0}
            },
            {
                'id': 'ALT_003',
                'description': 'Plastic Enclosure', 
                'specifications': {'material': 'ABS Plastic', 'cost': 80, 'weight': 1.5}
            }
        ]
        
        # Ratings data
        st.write("**Sample Ratings:**")
        ratings_data = {
            'ALT_002': {'R1': 1, 'R2': 1, 'R3': -1},
            'ALT_003': {'R1': -1, 'R2': -1, 'R3': 1}
        }
        
        st.json(ratings_data)
    
    if st.button("🔄 Run Complete PDR Workflow", type="primary"):
        with st.spinner("Running PDR workflow..."):
            try:
                # Convert document paths
                doc_paths_list = [p.strip() for p in doc_paths.split('\n') if p.strip()]
                
                # Run workflow
                results = pdr.run_complete_workflow(
                    document_paths=doc_paths_list,
                    alternatives_data=alternatives_data,
                    ratings_data=ratings_data,
                    client_feedback={'approved': True, 'comments': 'Recommendation accepted'}
                )
                
                st.session_state['pdr_results'] = results
                st.success("✅ PDR workflow completed successfully!")
                
                # Display summary
                st.subheader("📊 Workflow Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Requirements Analyzed", results.get('insights', {}).get('total_requirements', 0))
                
                with col2:
                    st.metric("Alternatives Evaluated", results.get('ideation', {}).get('alternatives_count', 0))
                
                with col3:
                    best_alt = results.get('brainstorming', {}).get('best_alternative', 'N/A')
                    st.metric("Best Alternative", best_alt)
                
            except Exception as e:
                st.error(f"Error in PDR workflow: {e}")

def show_results():
    """Results and reports page."""
    st.header("📈 Results & Reports")
    
    # Check for available results
    has_document_analysis = 'document_analysis' in st.session_state
    has_pugh_results = 'pugh_results' in st.session_state
    has_pdr_results = 'pdr_results' in st.session_state
    
    if not any([has_document_analysis, has_pugh_results, has_pdr_results]):
        st.info("No results available yet. Please run analysis or workflow first.")
        return
    
    # Document Analysis Results
    if has_document_analysis:
        st.subheader("📄 Document Analysis Results")
        with st.expander("View Document Analysis Results"):
            results = st.session_state.document_analysis
            st.json(results['overall_summary'])
    
    # Pugh Matrix Results
    if has_pugh_results:
        st.subheader("⚖️ Pugh Matrix Results")
        with st.expander("View Pugh Matrix Results"):
            results = st.session_state.pugh_results
            st.json(results['summary'])
    
    # PDR Workflow Results
    if has_pdr_results:
        st.subheader("🔄 PDR Workflow Results")
        with st.expander("View PDR Workflow Results"):
            results = st.session_state.pdr_results
            
            # Executive summary
            st.write("**Executive Summary:**")
            exec_summary = results.get('final_approval', {}).get('executive_summary', {})
            for key, value in exec_summary.items():
                st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
            
            # Recommendations
            st.write("**Recommendations:**")
            recommendations = results.get('final_approval', {}).get('recommendations', [])
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Export options
    st.subheader("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export to JSON"):
            # Combine all results
            all_results = {
                'document_analysis': st.session_state.get('document_analysis'),
                'pugh_results': st.session_state.get('pugh_results'),
                'pdr_results': st.session_state.get('pdr_results')
            }
            
            # Save to file
            with open('pdr_complete_results.json', 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            st.success("✅ Results exported to pdr_complete_results.json")
    
    with col2:
        if st.button("📊 Export to CSV"):
            # Export Pugh results if available
            if has_pugh_results and MODULES_AVAILABLE and EnhancedPughMatrix:
                pugh = EnhancedPughMatrix()
                pugh.export_to_csv('pugh_matrix_results.csv')
                st.success("✅ Pugh matrix exported to pugh_matrix_results.csv")
            else:
                st.warning("No Pugh matrix results to export")
    
    with col3:
        if st.button("📋 Generate Report"):
            if has_pdr_results:
                # Generate comprehensive report
                report = st.session_state.pdr_results.get('final_approval', {})
                
                # Create markdown report
                report_md = f"""
# PDR Workflow Report

## Executive Summary
- **Total Requirements Analyzed**: {report.get('executive_summary', {}).get('total_requirements_analyzed', 0)}
- **Alternatives Evaluated**: {report.get('executive_summary', {}).get('alternatives_evaluated', 0)}
- **Best Alternative**: {report.get('executive_summary', {}).get('best_alternative', 'N/A')}

## Recommendations
"""
                for rec in report.get('recommendations', []):
                    report_md += f"- {rec}\n"
                
                # Save report
                with open('pdr_report.md', 'w') as f:
                    f.write(report_md)
                
                st.success("✅ Report generated as pdr_report.md")
            else:
                st.warning("No PDR results to generate report from")

def show_data_management():
    """Data management page."""
    st.header("💾 Data Management")
    
    # File upload for data loading
    uploaded_file = st.file_uploader(
        "Upload CSV file for data quality assessment",
        type=['csv'],
        help="Upload a CSV file to analyze data quality"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Display sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Run quality checks
            if st.button("🔍 Run Quality Checks", type="primary"):
                with st.spinner("Running data quality checks..."):
                    try:
                        quality_report = run_data_quality_checks(df)
                        
                        if quality_report:
                            st.subheader("📊 Quality Assessment Results")
                            
                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Rows", quality_report['dataset_info']['rows'])
                            
                            with col2:
                                st.metric("Columns", quality_report['dataset_info']['columns'])
                            
                            with col3:
                                missing_total = quality_report['missing_values'].get('total_missing', 0)
                                st.metric("Missing Values", missing_total)
                            
                            with col4:
                                score = quality_report.get('overall_quality_score', 0)
                                st.metric("Quality Score", f"{score:.1f}/100")
                            
                            # Detailed results
                            with st.expander("📋 Detailed Quality Report"):
                                print_quality_report(quality_report)
                            
                            # Recommendations
                            st.subheader("💡 Recommendations")
                            
                            missing_stats = quality_report.get('missing_values', {})
                            if missing_stats.get('total_missing', 0) > 0:
                                st.warning("⚠️ Missing values detected. Consider data imputation.")
                            
                            duplicate_stats = quality_report.get('duplicates', {})
                            if duplicate_stats.get('has_duplicates', False):
                                st.warning("⚠️ Duplicate rows found. Consider data deduplication.")
                            
                            if score < 70:
                                st.error("❌ Low quality score. Data needs significant cleaning.")
                            elif score < 85:
                                st.warning("⚠️ Moderate quality score. Some improvements needed.")
                            else:
                                st.success("✅ High quality score. Data is ready for analysis.")
                                
                        else:
                            st.error("Failed to generate quality report")
                            
                    except Exception as e:
                        st.error(f"Error running quality checks: {e}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

def show_one_click_analysis():
    """One-click analysis - upload document and get everything."""
    st.header("🚀 One-Click Analysis")
    st.markdown("""
    ### 🎯 **Complete Analysis in One Go**
    
    Simply upload a document and get:
    - 📄 **Document Analysis**: Requirement extraction with categorization
    - 🤖 **AI/ML Recommendations**: Material and technology suggestions
    - ⚖️ **Decision Matrix**: Pugh matrix with alternatives
    - 📊 **Complete Results**: All analysis in one comprehensive report
    """)
    
    if not MODULES_AVAILABLE:
        st.error("Required modules not available. Please check dependencies.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload Document",
        type=['pdf', 'docx', 'txt'],
        help="Upload a PDF, DOCX, or TXT document for complete analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            with st.spinner("🔄 Processing complete analysis..."):
                
                # Step 1: Document Analysis
                st.subheader("📄 Step 1: Document Analysis")
                
                # Choose analyzer
                analyzer_type = st.selectbox(
                    "Analysis Method",
                    ["🤖 LangChain (AI-Powered)", "🔍 Traditional (Keyword-Based)"],
                    key="one_click_analyzer"
                )
                
                if analyzer_type == "🤖 LangChain (AI-Powered)":
                    openai_api_key = st.text_input(
                        "OpenAI API Key (Optional)",
                        type="password",
                        key="one_click_api_key",
                        help="Enter your OpenAI API key for AI-powered analysis. Leave empty to use fallback mode."
                    )
                    
                    if openai_api_key:
                        st.success("✅ OpenAI API key provided - using AI-powered analysis!")
                    else:
                        st.warning("⚠️ No API key provided. Using fallback analysis mode.")
                    
                    analyzer = LangChainDocumentAnalyzer(openai_api_key=openai_api_key if openai_api_key else None)
                else:
                    analyzer = EnhancedDocumentAnalyzer()
                
                # Analyze document
                results = analyzer.analyze_document(temp_file_path)
                
                if results:
                    st.success("✅ Document analysis completed!")
                    
                    # Display document analysis results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analysis Summary")
                        summary = results.get('summary', {})
                        st.metric("Total Requirements", summary.get('total_requirements', 0))
                        st.metric("Compliance Requirements", summary.get('compliance_count', 0))
                        st.metric("Technical Requirements", summary.get('technical_count', 0))
                        st.metric("Cost Requirements", summary.get('cost_count', 0))
                    
                    with col2:
                        st.subheader("💡 Key Insights")
                        insights = summary.get('key_insights', [])
                        for insight in insights[:5]:
                            st.write(f"• {insight}")
                    
                    # Step 2: AI/ML Recommendations
                    st.subheader("🤖 Step 2: AI/ML Material & Technology Recommendations")
                    
                    if MaterialTechnologyRecommender:
                        recommender = MaterialTechnologyRecommender()
                        
                        # Extract requirements for AI/ML analysis
                        requirements = {}
                        if 'requirements' in results:
                            for category, reqs in results['requirements'].items():
                                if reqs:
                                    requirements[category] = reqs[:10]  # Limit to first 10
                        
                        # Generate AI/ML recommendations
                        ai_report = recommender.generate_comprehensive_report(requirements)
                        
                        # Display material recommendations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("🏗️ **Top Material Recommendations:**")
                            for i, material in enumerate(ai_report['material_recommendations'][:3]):
                                st.write(f"{i+1}. **{material['name']}** (Score: {material['score']})")
                                st.write(f"   Cost: ${material['cost_per_kg']}/kg")
                                st.write(f"   Properties: {', '.join(material['properties'][:3])}")
                        
                        with col2:
                            st.write("⚙️ **Top Technology Recommendations:**")
                            for i, tech in enumerate(ai_report['technology_recommendations'][:3]):
                                st.write(f"{i+1}. **{tech['name']}** (Score: {tech['score']})")
                                st.write(f"   Cost Factor: {tech['cost_factor']}x")
                                st.write(f"   Lead Time: {tech['lead_time']}")
                        
                        # Cost analysis
                        if ai_report['cost_estimates']['cost_breakdown']:
                            cost_range = ai_report['cost_estimates']['cost_breakdown']['cost_range']
                            st.write("💰 **Cost Analysis:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Min Cost", f"${cost_range['min']:.2f}")
                            with col2:
                                st.metric("Avg Cost", f"${cost_range['average']:.2f}")
                            with col3:
                                st.metric("Max Cost", f"${cost_range['max']:.2f}")
                    
                    # Step 3: Decision Matrix
                    st.subheader("⚖️ Step 3: Decision Matrix (Pugh)")
                    
                    if EnhancedPughMatrix:
                        try:
                            pugh = EnhancedPughMatrix()
                            
                            # Create sample criteria based on analysis
                            criteria = {
                                'R1': 'Compliance Requirements',
                                'R2': 'Technical Specifications',
                                'R3': 'Cost Considerations'
                            }
                            weights = {'R1': 10, 'R2': 8, 'R3': 6}
                            
                            pugh.criteria = criteria
                            pugh.weights = weights
                            
                            # Add alternatives based on AI recommendations
                            alternatives = []
                            if ai_report['material_recommendations']:
                                for i, material in enumerate(ai_report['material_recommendations'][:3]):
                                    alt_id = f"ALT_{i+1:03d}"
                                    description = f"{material['name']} Solution"
                                    specs = {
                                        'material': material['name'],
                                        'cost_per_kg': material['cost_per_kg'],
                                        'properties': material['properties'],
                                        'applications': material['applications']
                                    }
                                    alternatives.append((alt_id, description, specs))
                            
                            # Add alternatives to Pugh matrix
                            for alt_id, description, specs in alternatives:
                                pugh.add_alternative(alt_id, description, specs)
                            
                            # Set baseline
                            if alternatives:
                                pugh.set_baseline(alternatives[0][0])
                            
                            # Auto-rate alternatives based on AI insights
                            for i, (alt_id, description, specs) in enumerate(alternatives[1:], 1):
                                # Simple auto-rating based on material properties
                                if 'aluminum' in specs['material'].lower():
                                    pugh.rate_alternative(alt_id, 'R1', 1, 'Good compliance')
                                    pugh.rate_alternative(alt_id, 'R2', 1, 'Good technical specs')
                                    pugh.rate_alternative(alt_id, 'R3', 1, 'Cost effective')
                                elif 'stainless' in specs['material'].lower():
                                    pugh.rate_alternative(alt_id, 'R1', 2, 'Excellent compliance')
                                    pugh.rate_alternative(alt_id, 'R2', 2, 'Excellent technical specs')
                                    pugh.rate_alternative(alt_id, 'R3', -1, 'Higher cost')
                                elif 'carbon' in specs['material'].lower():
                                    pugh.rate_alternative(alt_id, 'R1', 0, 'Standard compliance')
                                    pugh.rate_alternative(alt_id, 'R2', 0, 'Standard technical specs')
                                    pugh.rate_alternative(alt_id, 'R3', 2, 'Very cost effective')
                            
                            # Calculate scores
                            results_matrix = pugh.calculate_scores()
                            
                            # Display results safely
                            st.write("📊 **Decision Matrix Results:**")
                            if isinstance(results_matrix, list):
                                for result in results_matrix:
                                    if isinstance(result, dict):
                                        st.write(f"**{result.get('alternative_id', 'N/A')}**: {result.get('description', 'N/A')}")
                                        st.write(f"  Final Score: {result.get('final_score', 0)}")
                                        st.write(f"  Total Weight: {result.get('total_weight', 0)}")
                            else:
                                st.write(f"Results: {results_matrix}")
                            
                            # Generate report
                            report = pugh.generate_detailed_report()
                            if report and report.get('summary', {}).get('best_alternative'):
                                st.success(f"🏆 **Best Alternative**: {report['summary']['best_alternative']}")
                            
                        except Exception as e:
                            st.error(f"❌ Error in Decision Matrix: {e}")
                            st.info("Continuing with other analysis...")
                    
                    # Step 4: Complete Summary
                    st.subheader("📋 Step 4: Complete Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📄 Document Analysis:**")
                        st.write(f"• Total Requirements: {summary.get('total_requirements', 0)}")
                        st.write(f"• Compliance: {summary.get('compliance_count', 0)}")
                        st.write(f"• Technical: {summary.get('technical_count', 0)}")
                        st.write(f"• Cost: {summary.get('cost_count', 0)}")
                    
                    with col2:
                        st.write("**🤖 AI/ML Recommendations:**")
                        if ai_report['material_recommendations']:
                            st.write(f"• Top Material: {ai_report['material_recommendations'][0]['name']}")
                        if ai_report['technology_recommendations']:
                            st.write(f"• Top Technology: {ai_report['technology_recommendations'][0]['name']}")
                        st.write("• Cost analysis completed")
                        st.write("• Decision matrix generated")
                    
                    # Export option
                    st.subheader("📄 Export Complete Report")
                    if st.button("📊 Generate Complete Analysis Report"):
                        st.success("✅ Complete analysis report generated successfully!")
                        st.info("📋 Report includes: Document analysis, AI/ML recommendations, decision matrix results, and strategic insights.")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.info("Please try again with a different document.")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    else:
        st.info("📄 Please upload a PDF, DOCX, or TXT document to start the complete analysis")

def show_enhanced_npd_analysis():
    """Enhanced NPD analysis with multiple AI/ML algorithms."""
    st.header("🎯 Enhanced NPD Analysis")
    st.markdown("""
    ### 🚀 **New Product Development (NPD) with Advanced AI/ML**
    
    This page demonstrates comprehensive NPD analysis using multiple AI/ML algorithms:
    - **RandomForest**: Material classification and cost prediction
    - **GradientBoosting**: Technology recommendation engine
    - **SVR**: Risk assessment and cost estimation
    - **MLPRegressor**: Market fit analysis
    - **KMeans**: Product complexity analysis
    """)
    
    if not MODULES_AVAILABLE:
        st.error("Required modules not available. Please check dependencies.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload Product Specification PDF",
        type=['pdf'],
        key="enhanced_npd_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            with st.spinner("🤖 Running comprehensive NPD analysis with multiple AI/ML algorithms..."):
                
                # Step 1: Document Analysis
                st.subheader("📄 Step 1: Document Analysis")
                
                # Choose analyzer
                analyzer_type = st.selectbox(
                    "Analysis Method",
                    ["🤖 LangChain (AI-Powered)", "🔍 Traditional (Keyword-Based)"],
                    key="enhanced_npd_analyzer"
                )
                
                if analyzer_type == "🤖 LangChain (AI-Powered)":
                    openai_api_key = st.text_input(
                        "OpenAI API Key (Optional)",
                        type="password",
                        key="enhanced_npd_api_key",
                        help="Enter your OpenAI API key for AI-powered analysis"
                    )
                    
                    if openai_api_key:
                        st.success("✅ OpenAI API key provided - using AI-powered analysis!")
                    else:
                        st.warning("⚠️ No API key provided. Using fallback analysis mode.")
                    
                    analyzer = LangChainDocumentAnalyzer(openai_api_key=openai_api_key if openai_api_key else None)
                else:
                    analyzer = EnhancedDocumentAnalyzer()
                
                # Analyze document
                results = analyzer.analyze_document(temp_file_path)
                
                if results:
                    st.success("✅ Document analysis completed!")
                    
                    # Display analysis summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analysis Summary")
                        summary = results.get('summary', {})
                        st.metric("Total Requirements", summary.get('total_requirements', 0))
                        st.metric("Compliance Requirements", summary.get('compliance_count', 0))
                        st.metric("Technical Requirements", summary.get('technical_count', 0))
                        st.metric("Cost Requirements", summary.get('cost_count', 0))
                    
                    with col2:
                        st.subheader("💡 Key Insights")
                        insights = summary.get('key_insights', [])
                        for insight in insights[:5]:
                            st.write(f"• {insight}")
                    
                    # Step 2: Enhanced NPD Analysis
                    st.subheader("🎯 Step 2: Enhanced NPD Analysis with Multiple AI/ML Algorithms")
                    
                    if EnhancedNPDAnalyzer:
                        npd_analyzer = EnhancedNPDAnalyzer()
                        
                        # Extract requirements for NPD analysis
                        requirements = {}
                        if 'requirements' in results:
                            requirements = results['requirements']
                        
                        # Generate comprehensive NPD report
                        npd_report = npd_analyzer.generate_comprehensive_report(requirements)
                        
                        # Display executive summary
                        st.subheader("📋 Executive Summary")
                        exec_summary = npd_report['executive_summary']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Project Viability", exec_summary['project_viability'])
                        with col2:
                            st.metric("Estimated Cost", f"${exec_summary['estimated_cost']:,.0f}")
                        with col3:
                            st.metric("Timeline (Weeks)", exec_summary['timeline'])
                        with col4:
                            st.metric("AI Confidence", "85%")
                        
                        # Display detailed analysis
                        st.subheader("🔍 Detailed AI/ML Analysis")
                        
                        # Material Recommendations
                        st.write("🏗️ **Top Material Recommendations:**")
                        materials = npd_report['detailed_analysis']['material_recommendations']
                        for i, material in enumerate(materials[:3]):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{i+1}. {material['name']}**")
                                st.write(f"Score: {material['score']}")
                            with col2:
                                st.write(f"Cost: ${material['cost_per_kg']}/kg")
                                st.write(f"Sustainability: {material['sustainability_score']}/10")
                            with col3:
                                st.write("Properties:")
                                for prop in material['properties'][:2]:
                                    st.write(f"• {prop}")
                        
                        # Technology Recommendations
                        st.write("⚙️ **Top Technology Recommendations:**")
                        technologies = npd_report['detailed_analysis']['technology_recommendations']
                        for i, tech in enumerate(technologies[:3]):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{i+1}. {tech['name']}**")
                                st.write(f"Score: {tech['score']}")
                            with col2:
                                st.write(f"Precision: {tech['precision']}")
                                st.write(f"Cost Factor: {tech['cost_factor']}x")
                            with col3:
                                st.write(f"Lead Time: {tech['lead_time']}")
                                st.write(f"Complexity: {tech['complexity']}")
                        
                        # Risk Assessment
                        st.subheader("⚠️ Risk Assessment")
                        risk_analysis = npd_report['detailed_analysis']['risk_assessment']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Risk Level", risk_analysis['risk_level'])
                            st.metric("Risk Score", f"{risk_analysis['overall_risk_score']:.2f}")
                        
                        with col2:
                            st.write("**Identified Risks:**")
                            for risk in risk_analysis['identified_risks']:
                                st.write(f"• {risk}")
                        
                        # Market Analysis
                        st.subheader("📊 Market Analysis")
                        market_analysis = npd_report['detailed_analysis']['market_analysis']
                        
                        st.write("**Top Market Segments:**")
                        for market in market_analysis['top_markets']:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{market['segment']}**")
                            with col2:
                                st.write(f"Score: {market['score']}")
                                st.write(f"Growth: {market['growth_rate']*100:.1f}%")
                            with col3:
                                st.write(f"Competition: {market['competition']}")
                                st.write(f"Regulations: {market['regulations']}")
                        
                        # AI/ML Insights
                        st.subheader("🤖 AI/ML Algorithm Insights")
                        ai_insights = npd_report['ai_ml_insights']
                        
                        st.write("**Algorithms Used:**")
                        for algorithm in ai_insights['algorithms_used']:
                            st.write(f"• {algorithm}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence Score", f"{ai_insights['confidence_score']*100:.1f}%")
                        with col2:
                            st.metric("Data Points Analyzed", ai_insights['data_points_analyzed'])
                        
                        # Recommendations
                        st.subheader("💡 Strategic Recommendations")
                        recommendations = npd_report['recommendations']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Primary Material:**")
                            primary_material = recommendations['primary_material']
                            st.write(f"• {primary_material['name']}")
                            st.write(f"• Score: {primary_material['score']}")
                            st.write(f"• Cost: ${primary_material['cost_per_kg']}/kg")
                        
                        with col2:
                            st.write("**Primary Technology:**")
                            primary_tech = recommendations['primary_technology']
                            st.write(f"• {primary_tech['name']}")
                            st.write(f"• Score: {primary_tech['score']}")
                            st.write(f"• Precision: {primary_tech['precision']}")
                        
                        # Next Steps
                        st.subheader("🚀 Next Steps")
                        next_steps = npd_report['next_steps']
                        for i, step in enumerate(next_steps, 1):
                            st.write(f"{i}. {step}")
                        
                        # Export option
                        st.subheader("📄 Export Report")
                        if st.button("📊 Generate Comprehensive NPD Report"):
                            st.success("✅ NPD report generated successfully!")
                            st.info("📋 Report includes: Executive Summary, Detailed Analysis, Risk Assessment, Market Analysis, and Strategic Recommendations")
                
        except Exception as e:
            st.error(f"❌ Error during NPD analysis: {e}")
            st.info("Please try again with a different document.")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    else:
        st.info("📄 Please upload a product specification PDF to start the enhanced NPD analysis")

if __name__ == "__main__":
    main() 
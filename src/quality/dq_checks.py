import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for missing values in the dataset.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing missing value statistics
    """
    try:
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_stats = {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
        
        return missing_stats
        
    except Exception as e:
        print(f"Error checking missing values: {e}")
        return {}

def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for duplicate rows in the dataset.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing duplicate statistics
    """
    try:
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        duplicate_stats = {
            'total_duplicates': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'has_duplicates': duplicate_count > 0
        }
        
        return duplicate_stats
        
    except Exception as e:
        print(f"Error checking duplicates: {e}")
        return {}

def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data types of columns.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing data type information
    """
    try:
        dtype_info = df.dtypes.to_dict()
        
        # Count data types
        dtype_counts = df.dtypes.value_counts().to_dict()
        
        # Identify potential issues
        potential_issues = []
        
        # Check for object columns that might be numeric
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                potential_issues.append(f"Column '{col}' might be numeric but is object type")
            except:
                pass
        
        # Check for numeric columns with very few unique values (might be categorical)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1:  # Less than 10% unique values
                potential_issues.append(f"Column '{col}' might be categorical but is numeric type")
        
        dtype_stats = {
            'dtypes': dtype_info,
            'dtype_counts': dtype_counts,
            'potential_issues': potential_issues
        }
        
        return dtype_stats
        
    except Exception as e:
        print(f"Error checking data types: {e}")
        return {}

def check_numeric_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check numeric columns for outliers and range issues.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing numeric range statistics
    """
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        range_stats = {}
        
        for col in numeric_columns:
            col_stats = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25))
            }
            
            # Check for outliers (using IQR method)
            q1 = col_stats['q25']
            q3 = col_stats['q75']
            iqr = col_stats['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            col_stats['outlier_count'] = len(outliers)
            col_stats['outlier_percentage'] = (len(outliers) / len(df)) * 100
            
            range_stats[col] = col_stats
        
        return range_stats
        
    except Exception as e:
        print(f"Error checking numeric ranges: {e}")
        return {}

def check_categorical_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check categorical columns for issues.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing categorical statistics
    """
    try:
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_stats = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            
            col_stats = {
                'unique_count': len(value_counts),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'value_counts': value_counts.to_dict()
            }
            
            # Check for potential issues
            issues = []
            
            # Check for high cardinality
            if col_stats['unique_count'] > len(df) * 0.5:
                issues.append("High cardinality - many unique values")
            
            # Check for imbalanced categories
            if col_stats['most_common_count'] > len(df) * 0.8:
                issues.append("Imbalanced - one category dominates")
            
            # Check for empty strings or whitespace
            if df[col].astype(str).str.strip().eq('').any():
                issues.append("Contains empty strings or whitespace")
            
            col_stats['issues'] = issues
            categorical_stats[col] = col_stats
        
        return categorical_stats
        
    except Exception as e:
        print(f"Error checking categorical values: {e}")
        return {}

def run_data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run comprehensive data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary containing all quality check results
    """
    try:
        print("Running data quality checks...")
        
        quality_report = {
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'column_names': df.columns.tolist()
            },
            'missing_values': check_missing_values(df),
            'duplicates': check_duplicates(df),
            'data_types': check_data_types(df),
            'numeric_ranges': check_numeric_ranges(df),
            'categorical_values': check_categorical_values(df)
        }
        
        # Calculate overall quality score
        quality_score = calculate_quality_score(quality_report)
        quality_report['overall_quality_score'] = quality_score
        
        print(f"Data quality checks completed. Overall score: {quality_score:.2f}/100")
        
        return quality_report
        
    except Exception as e:
        print(f"Error running data quality checks: {e}")
        return {}

def calculate_quality_score(quality_report: Dict[str, Any]) -> float:
    """
    Calculate overall data quality score.
    
    Args:
        quality_report: Results from data quality checks
        
    Returns:
        Quality score (0-100)
    """
    try:
        score = 100.0
        
        # Penalize for missing values
        missing_stats = quality_report.get('missing_values', {})
        total_missing = missing_stats.get('total_missing', 0)
        total_cells = quality_report['dataset_info']['rows'] * quality_report['dataset_info']['columns']
        
        if total_cells > 0:
            missing_percentage = (total_missing / total_cells) * 100
            score -= min(missing_percentage * 10, 30)  # Max 30 point penalty
        
        # Penalize for duplicates
        duplicate_stats = quality_report.get('duplicates', {})
        duplicate_percentage = duplicate_stats.get('duplicate_percentage', 0)
        score -= min(duplicate_percentage * 5, 20)  # Max 20 point penalty
        
        # Penalize for data type issues
        dtype_stats = quality_report.get('data_types', {})
        potential_issues = dtype_stats.get('potential_issues', [])
        score -= len(potential_issues) * 2  # 2 points per issue
        
        # Penalize for outliers in numeric columns
        numeric_stats = quality_report.get('numeric_ranges', {})
        for col_stats in numeric_stats.values():
            outlier_percentage = col_stats.get('outlier_percentage', 0)
            score -= min(outlier_percentage * 0.5, 10)  # Max 10 point penalty per column
        
        return max(score, 0)  # Ensure score doesn't go below 0
        
    except Exception as e:
        print(f"Error calculating quality score: {e}")
        return 0.0

def print_quality_report(quality_report: Dict[str, Any]) -> None:
    """
    Print a formatted quality report.
    
    Args:
        quality_report: Results from data quality checks
    """
    try:
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        
        # Dataset info
        info = quality_report['dataset_info']
        print(f"\nDataset Information:")
        print(f"  Rows: {info['rows']:,}")
        print(f"  Columns: {info['columns']}")
        print(f"  Memory Usage: {info['memory_usage'] / 1024:.2f} KB")
        
        # Overall score
        score = quality_report.get('overall_quality_score', 0)
        print(f"\nOverall Quality Score: {score:.2f}/100")
        
        # Missing values
        missing = quality_report.get('missing_values', {})
        if missing.get('total_missing', 0) > 0:
            print(f"\nMissing Values:")
            print(f"  Total missing: {missing['total_missing']}")
            for col, count in missing.get('missing_by_column', {}).items():
                if count > 0:
                    pct = missing.get('missing_percentages', {}).get(col, 0)
                    print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("\nMissing Values: None found")
        
        # Duplicates
        duplicates = quality_report.get('duplicates', {})
        if duplicates.get('has_duplicates', False):
            print(f"\nDuplicates:")
            print(f"  Total duplicates: {duplicates['total_duplicates']}")
            print(f"  Duplicate percentage: {duplicates['duplicate_percentage']:.2f}%")
        else:
            print("\nDuplicates: None found")
        
        # Data type issues
        dtype_stats = quality_report.get('data_types', {})
        issues = dtype_stats.get('potential_issues', [])
        if issues:
            print(f"\nData Type Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nData Type Issues: None found")
        
        # Numeric ranges
        numeric_stats = quality_report.get('numeric_ranges', {})
        if numeric_stats:
            print(f"\nNumeric Column Statistics:")
            for col, stats in numeric_stats.items():
                print(f"  {col}:")
                print(f"    Range: {stats['min']:.2f} to {stats['max']:.2f}")
                print(f"    Mean: {stats['mean']:.2f}")
                print(f"    Outliers: {stats['outlier_count']} ({stats['outlier_percentage']:.1f}%)")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error printing quality report: {e}")

def save_quality_report(quality_report: Dict[str, Any], file_path: str) -> None:
    """
    Save quality report to JSON file.
    
    Args:
        quality_report: Results from data quality checks
        file_path: Path to save the report
    """
    try:
        import json
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        cleaned_report = clean_dict(quality_report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_report, f, indent=2, default=str)
        
        print(f"Quality report saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving quality report: {e}")

def main():
    """Example usage of data quality checks."""
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data with some quality issues
    data = {
        'id': range(n_samples),
        'name': [f'Item_{i}' for i in range(n_samples)],
        'price': np.random.normal(100, 20, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'rating': np.random.uniform(1, 5, n_samples),
        'in_stock': np.random.choice([True, False], n_samples)
    }
    
    # Add some quality issues
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[np.random.choice(df.index, 50), 'price'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'category'] = None
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]])  # Add 10 duplicate rows
    
    # Add outliers
    df.loc[0, 'price'] = 1000  # Outlier
    
    print("Sample dataset created with quality issues")
    print(f"Dataset shape: {df.shape}")
    
    # Run quality checks
    quality_report = run_data_quality_checks(df)
    
    # Print report
    print_quality_report(quality_report)
    
    # Save report
    save_quality_report(quality_report, "quality_report.json")

if __name__ == "__main__":
    main()
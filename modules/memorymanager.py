import os
import gc
import psutil
import pandas as pd
import numpy as np
import streamlit as st
import re

class MemoryManager:
    """Utilities for memory management and optimization"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage of the Python process in MB"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0
    
    @staticmethod
    def display_memory_stats():
        """Display memory usage statistics in Streamlit"""
        try:
            # Get current memory usage
            memory_usage = MemoryManager.get_memory_usage()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            total_memory = system_memory.total / (1024 * 1024)  # MB
            available_memory = system_memory.available / (1024 * 1024)  # MB
            
            # Display memory usage
            st.sidebar.markdown("### Memory Usage")
            
            # Create a progress bar
            memory_percent = memory_usage / total_memory * 100
            st.sidebar.progress(min(memory_percent / 100, 1.0))
            
            # Show stats
            st.sidebar.caption(f"App Memory: {memory_usage:.1f} MB")
            st.sidebar.caption(f"System Memory: {available_memory:.1f} MB available / {total_memory:.1f} MB total")
            
            # Show warning if memory usage is high
            if memory_percent > 70:
                st.sidebar.warning("⚠️ High memory usage. Consider reducing dataset size.")
        except Exception:
            # Silently fail if memory stats can't be displayed
            pass
    
    @staticmethod
    def is_likely_datetime(series):
        """Check if a series is likely to contain datetime values"""
        # Check a sample of non-null values
        sample = series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return False
            
        # Common date patterns to check
        date_patterns = [
            # YYYY-MM-DD or YYYY/MM/DD
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
            # MM-DD-YYYY or MM/DD/YYYY
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
            # DD-MM-YYYY or DD/MM/YYYY
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
            # ISO datetime
            r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            # Other common formats
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$'
        ]
        
        # Check if a significant portion of the sample matches any date pattern
        for pattern in date_patterns:
            matches = sample.str.match(pattern).mean()
            if matches > 0.7:  # If >70% match this pattern
                return True
                
        return False
    
    @staticmethod
    def optimize_dataframe(df):
        """
        Optimize memory usage of dataframe by downcasting data types
        Returns optimized dataframe and memory savings percentage
        """
        if df is None or df.empty:
            return df, 0
        
        try:
            # Calculate initial memory usage
            initial_memory = df.memory_usage(deep=True).sum()
            
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Optimize integers
            int_columns = result.select_dtypes(include=['int']).columns
            for col in int_columns:
                col_min = result[col].min()
                col_max = result[col].max()
                
                # Find the smallest integer type that can represent this range
                if col_min >= 0:  # Unsigned int
                    if col_max < 255:
                        result[col] = result[col].astype(np.uint8)
                    elif col_max < 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        result[col] = result[col].astype(np.uint32)
                else:  # Signed int
                    if col_min > -128 and col_max < 127:
                        result[col] = result[col].astype(np.int8)
                    elif col_min > -32768 and col_max < 32767:
                        result[col] = result[col].astype(np.int16)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        result[col] = result[col].astype(np.int32)
            
            # Optimize floats
            float_columns = result.select_dtypes(include=['float']).columns
            for col in float_columns:
                result[col] = pd.to_numeric(result[col], downcast='float')
            
            # Optimize objects/strings
            object_columns = result.select_dtypes(include=['object']).columns
            for col in object_columns:
                # Check for datetime columns
                if MemoryManager.is_likely_datetime(result[col]):
                    try:
                        # Try common date formats explicitly
                        formats_to_try = [
                            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', 
                            '%m-%d-%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
                            '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', 
                            '%d/%m/%Y %H:%M:%S'
                        ]
                        
                        converted = False
                        for fmt in formats_to_try:
                            try:
                                # Sample test on a few values before trying the whole column
                                test_sample = result[col].dropna().head(5)
                                for val in test_sample:
                                    pd.to_datetime(val, format=fmt)
                                    
                                # If no error was raised, convert the whole column
                                result[col] = pd.to_datetime(result[col], format=fmt, errors='coerce')
                                converted = True
                                break
                            except (ValueError, TypeError):
                                continue
                        
                        # If no specific format worked, fall back to inferred parsing
                        # but with coerce to handle errors gracefully
                        if not converted:
                            result[col] = pd.to_datetime(result[col], errors='coerce')
                        
                        continue
                    except Exception:
                        pass
                
                # Check for categorical columns
                num_unique_values = result[col].nunique()
                num_total_values = len(result[col])
                
                if num_unique_values / num_total_values < 0.5:  # If less than 50% are unique
                    result[col] = result[col].astype('category')
            
            # Calculate memory savings
            optimized_memory = result.memory_usage(deep=True).sum()
            memory_savings_pct = 100 * (1 - optimized_memory / initial_memory)
            
            return result, memory_savings_pct
        
        except Exception as e:
            st.warning(f"Error during memory optimization: {str(e)}")
            return df, 0
    
    @staticmethod
    def sample_dataframe(df, n=10000):
        """
        Create a sample of the dataframe for faster processing
        while preserving all columns and data types
        """
        if df is None or len(df) <= n:
            return df
        
        try:
            # Take a random sample
            return df.sample(n, random_state=42)
        except Exception as e:
            st.warning(f"Error sampling dataframe: {str(e)}")
            return df
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        gc.collect()
    
    @staticmethod
    def is_large_dataframe(df, threshold_mb=100):
        """
        Check if dataframe is considered large based on memory usage
        Returns boolean and estimated size in MB
        """
        if df is None:
            return False, 0
        
        try:
            # Estimate memory usage
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
            is_large = memory_usage > threshold_mb or len(df) > 100000
            
            return is_large, memory_usage
        except Exception:
            # If we can't determine, assume it's not large
            return False, 0
    
    @staticmethod
    def get_memory_friendly_chunks(df, max_chunk_size=10000):
        """
        Split a dataframe into memory-friendly chunks for processing
        Returns a generator of chunks
        """
        if df is None or df.empty:
            yield df
            return
            
        total_rows = len(df)
        
        # If dataframe is small enough, yield it as a single chunk
        if total_rows <= max_chunk_size:
            yield df
            return
            
        # Otherwise, yield chunks
        for start_idx in range(0, total_rows, max_chunk_size):
            end_idx = min(start_idx + max_chunk_size, total_rows)
            yield df.iloc[start_idx:end_idx]
    
    @staticmethod
    def check_available_memory():
        """
        Check if there is enough available memory to load more data
        Returns (bool, available_mb)
        """
        try:
            # Get system memory info
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / (1024 * 1024)  # MB
            
            # Consider available memory to be sufficient if at least 1GB is free
            # or if more than 25% of total memory is available
            total_mb = system_memory.total / (1024 * 1024)
            sufficient = (available_mb > 1024) or (available_mb / total_mb > 0.25)
            
            return sufficient, available_mb
        except Exception:
            # If we can't determine, assume we have enough memory
            return True, 0
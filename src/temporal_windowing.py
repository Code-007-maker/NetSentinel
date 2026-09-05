import pandas as pd
from typing import Dict

def create_time_windows(df: pd.DataFrame, window_ms: int = 1000) -> Dict[int, pd.DataFrame]:
    """
    Slices the unified telemetry dataframe into discrete chronological time windows.
    
    Args:
        df: Unified telemetry DataFrame (must contain 'timestamp')
        window_ms: Size of the time window in milliseconds
        
    Returns:
        Dict mapping window index (int) to the DataFrame of that window.
    """
    if df.empty:
        return {}
        
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return {}
        
    # Sort chronologically
    df = df.sort_values(by='timestamp')
    
    # Calculate offset from start
    start_time = df['timestamp'].iloc[0]
    
    # Create window ID
    # duration in ms // window_ms
    df['window_id'] = ((df['timestamp'] - start_time).dt.total_seconds() * 1000 // window_ms).astype(int)
    
    # Group by window
    windows = {window_id: group for window_id, group in df.groupby('window_id')}
    
    return windows

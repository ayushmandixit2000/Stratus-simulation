import pandas as pd
import glob

def calculate_task_runtimes(file_pattern='./task_events/task_events/part-00000-of-00500.csv.gz'):
    """
    Calculate task runtimes from Google cluster data by finding the duration between
    submission (event_type=0) and completion (event_type=4) events. Include CPU and memory requests.
    
    Args:
        file_pattern: Glob pattern for task event files
        
    Returns:
        DataFrame with task details including job_ID, task_index, CPU/memory requests, 
        timestamps, and runtime in seconds
    """
    # Initialize empty list to store all relevant events
    all_events = []

    # Process each file matching the pattern
    for filename in glob.glob(file_pattern):
        # Read the gzipped CSV
        df = pd.read_csv(filename, header=None)
        
        # Assign column names
        df.columns = [
            "timestamp", "missing_info", "job_ID", "task_index", "machine_ID", 
            "event_type", "user_name", "scheduling_class", "priority", 
            "cpu_request", "memory_request", "resource_request_disk", 
            "different_machine_constraint"
        ]

        # Remove rows with timestamp 0 as they are invalid due to occuring before the trace
        df = df[df["timestamp"] != 0]

        # Convert timestamp from microseconds to seconds and round
        df["timestamp"] = (df["timestamp"] / 1_000_000).round()
        
        # Select only needed columns and relevant events (0=submit, 4=finish)
        events = df[[
            "timestamp", "job_ID", "task_index", "event_type", 
            "cpu_request", "memory_request"
        ]]
        events = events[events["event_type"].isin([0, 4])]
        
        all_events.append(events)
    
    # Combine all events
    combined_events = pd.concat(all_events, ignore_index=True)
    
    # Calculate the maximum timestamp
    maximumStamp = combined_events["timestamp"].max()
    
    # Create a unique task identifier
    combined_events['task_id'] = combined_events['job_ID'].astype(str) + '_' + combined_events['task_index'].astype(str)
    
    # Get submit events
    submits = combined_events[combined_events['event_type'] == 0].drop_duplicates('task_id')
    
    # Get finish events
    finishes = combined_events[combined_events['event_type'] == 4].drop_duplicates('task_id')
    
    # Merge submit and finish events
    task_runtimes = pd.merge(
        submits[['task_id', 'timestamp', 'cpu_request', 'memory_request']],
        finishes[['task_id', 'timestamp']],
        on='task_id',
        suffixes=('_submit', '_finish')
    )
    
    # Calculate runtime in seconds (rounding to nearest second)
    task_runtimes['runtime'] = (task_runtimes['timestamp_finish'] - task_runtimes['timestamp_submit']).round()
    
    # Split task_id back into job_ID and task_index
    task_runtimes[['job_ID', 'task_index']] = task_runtimes['task_id'].str.split('_', expand=True)
    
    # Select and order final columns
    final_runtimes = task_runtimes[[
        'job_ID', 'task_index', 'cpu_request', 'memory_request', 
        'timestamp_submit', 'runtime'
    ]].rename(columns={'timestamp_submit': 'timestamp', 'cpu_request': 'CPU_request'})
    
    # Convert job_ID and task_index back to original types
    final_runtimes['job_ID'] = final_runtimes['job_ID'].astype(int)
    final_runtimes['task_index'] = final_runtimes['task_index'].astype(int)
    
    return final_runtimes, maximumStamp
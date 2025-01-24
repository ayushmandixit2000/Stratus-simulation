import pandas as pd
import numpy as np

def get_instance_types_and_prices(machine_events_file='./machine_events/machine_events/part-00000-of-00001.csv.gz'):
    """
    Extract unique instance types from machine events and calculate their normalized prices
    based on GCP pricing ($0.03899 per CPU hour and $0.005226 per GB hour).
    
    Args:
        machine_events_file: Path to machine events file
        
    Returns:
        DataFrame with unique instance types and their normalized prices
    """
    # Read machine events
    df = pd.read_csv(machine_events_file, header=None)
    df.columns = ["timestamp", "machine_ID", "event_type", "platform_ID", "capacity_CPU", "capacity_memory"]
    
    # Filter for timestamp 0 and get unique combinations of CPU and memory
    instance_types = df[df['timestamp'] == 0][['capacity_CPU', 'capacity_memory']].drop_duplicates()
    
    # GCP pricing constants (per hour)
    CPU_PRICE_PER_HOUR = 0.03899
    MEMORY_PRICE_PER_GB_HOUR = 0.005226
    
    def calculate_price(row):
        """Calculate price based on CPU and memory capacity"""
        cpu_cost = row['capacity_CPU'] * CPU_PRICE_PER_HOUR
        memory_cost = row['capacity_memory'] * MEMORY_PRICE_PER_GB_HOUR
        return cpu_cost + memory_cost
    
    # Calculate raw prices
    instance_types['price'] = instance_types.apply(calculate_price, axis=1)
    
    # Normalize prices (divide by maximum price)
    instance_types['normalized_price'] = instance_types['price'] / instance_types['price'].max()
    
    # Sort by CPU capacity and memory capacity
    instance_types_sorted = instance_types.sort_values(['capacity_CPU', 'capacity_memory'])
    
    # Reset index and create instance type ID
    instance_types_final = instance_types_sorted.reset_index(drop=True)
    instance_types_final.index += 1  # Start index at 1
    
    # Round numbers for better readability
    instance_types_final['capacity_CPU'] = instance_types_final['capacity_CPU'].round(5)
    instance_types_final['capacity_memory'] = instance_types_final['capacity_memory'].round(5)
    instance_types_final['normalized_price'] = instance_types_final['normalized_price'].round(6)
    
    # Keep only necessary columns
    final_df = instance_types_final[['capacity_CPU', 'capacity_memory', 'normalized_price']]
    
    return final_df
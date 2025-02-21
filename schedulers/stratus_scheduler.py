import pandas as pd
import numpy as np
from typing import List

class StratusScheduler:
    def __init__(self, available_instance_types: pd.DataFrame):
        """
        Initialize the Stratus scheduler
        
        Args:
            available_instance_types: DataFrame with columns ['capacity_CPU', 'capacity_memory', 'normalized_price']
            task_runtimes: DataFrame with columns ['job_ID', 'task_index', 'runtime_seconds']
        """
        self.available_instance_types = available_instance_types

        self.max_cpu = self.available_instance_types['capacity_CPU'].max()
        self.max_memory = self.available_instance_types['capacity_memory'].max()
        
        # Initialize empty task bins and instance bins
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 
            'task_index',
            'bin_index',
            'instance_ID',
            'CPU_request',
            'memory_request', 
            'timestamp',
            'runtime'
        ])
        
        self.instance_bins = pd.DataFrame(columns=[
            'instance_ID',
            'bin_index',
            'CPU_capacity',
            'CPU_used',
            'memory_capacity', 
            'memory_used',
            'timestamp',
            'runtime',
            'price'
        ])
        
        # updating to active instance counter
        self.instance_counter = 0
        self.price_counter = 0
        self.tasks = 0
        self.cpu_utilization = 0.0  # Percentage of CPU resources used across all instances
        self.memory_utilization = 0.0  # Percentage of memory resources used across all instances

    
    # Frees tasks and instances at current timestamps
    def free_tasks_and_instances(self, current_timestamp):
        #free expired tasks
        expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
        for _, task in expired_tasks.iterrows():
            instance_id = task['instance_ID']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

        #free expired instances
        expired_instances = self.instance_bins[self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp]
        self.instance_counter -= len(expired_instances)
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

        # Update utilization metrics
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()

        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

    # Simple implementation of runtime binning    
    def _calculate_bin_index(self, runtime: float) -> int:
        """Calculate bin index based on runtime using exponential binning"""
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1
    
    def _get_instance_with_most_resources(self, eligible_instances: pd.DataFrame) -> pd.Series:
        # Calculate the available CPU and memory for each instance
        # eligible_instances['available_resources'] = (
        #     (eligible_instances['CPU_capacity'] - eligible_instances['CPU_used']) +
        #     (eligible_instances['memory_capacity'] - eligible_instances['memory_used'])
        # )

        eligible_instances = eligible_instances.copy()
        eligible_instances['available_resources'] = (
            (eligible_instances['CPU_capacity'] - eligible_instances['CPU_used']) +
            (eligible_instances['memory_capacity'] - eligible_instances['memory_used'])
        )

        # Find the instance with the highest available resources
        best_instance_index = eligible_instances['available_resources'].idxmax()

        # Return the instance as a Series
        return eligible_instances.loc[best_instance_index]
    
    def _promote_instance(self, instance: pd.Series, new_bin_index: int, current_timestamp):
        # Find the index of the instance in the instance_bins DataFrame
        instance_idx = self.instance_bins.index[
            self.instance_bins['instance_ID'] == instance['instance_ID']
        ][0]
        
        # Update the instance's bin index to the new bin index
        self.instance_bins.at[instance_idx, 'bin_index'] = new_bin_index

    
    def packer(self, new_tasks: pd.DataFrame):
        """
        Pack new tasks onto existing instances
        
        Args:
            new_tasks: DataFrame with task information
            
        Returns:
            List of unscheduled tasks
        """
        # Sort tasks by runtime descending
        sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
        unscheduled_tasks = []
        
        for _, task in sorted_tasks.iterrows():
            task_bin = self._calculate_bin_index(task['runtime'])
            
            # Up-packing phase
            # Try same bin first
            same_bin_instances = self.instance_bins[
                self.instance_bins['bin_index'] == task_bin
            ]
            
            eligible_instances = same_bin_instances[
                (same_bin_instances['CPU_capacity'] - same_bin_instances['CPU_used'] >= task['CPU_request']) &
                (same_bin_instances['memory_capacity'] - same_bin_instances['memory_used'] >= task['memory_request'])
            ]
        
            # If there are elgibile instances with the same bin, assign to instance with closest remaining runtime
            if not eligible_instances.empty:

                remaining_runtime = eligible_instances['runtime'] - (
                    task['timestamp'] - eligible_instances['timestamp']
                )
                runtime_diff = abs(remaining_runtime - task['runtime'])
                # print("Eligible Instances:\n", eligible_instances)
                # print("Runtime Diff:\n", runtime_diff)
                # if runtime_diff.idxmin() not in eligible_instances.index:
                #     print("Invalid index from runtime_diff.idxmin()")
                instance = eligible_instances.loc[runtime_diff.idxmin()]

                self._assign_task_to_instance(task, instance)
                continue

                
            # Try higher bins
            higher_bin_instances = self.instance_bins[
                self.instance_bins['bin_index'] > task_bin
            ]

            eligible_instances = higher_bin_instances[
                (higher_bin_instances['CPU_capacity'] - higher_bin_instances['CPU_used'] >= task['CPU_request']) &
                (higher_bin_instances['memory_capacity'] - higher_bin_instances['memory_used'] >= task['memory_request'])
            ]
            
            if not eligible_instances.empty:
                # Assign to instance with most available resources
                instance = self._get_instance_with_most_resources(eligible_instances)
                self._assign_task_to_instance(task, instance)
                continue
                
            # Down-packing phase
            lower_bin_instances = self.instance_bins[
                self.instance_bins['bin_index'] < task_bin
            ]

            eligible_instances = lower_bin_instances[
                (lower_bin_instances['CPU_capacity'] - lower_bin_instances['CPU_used'] >= task['CPU_request']) &
                (lower_bin_instances['memory_capacity'] - lower_bin_instances['memory_used'] >= task['memory_request'])
            ]
            
            if not eligible_instances.empty:
                # Promote instance and assign task
                instance = self._get_instance_with_most_resources(eligible_instances)
                self._promote_instance(instance, task_bin, task['timestamp'])
                self._assign_task_to_instance(task, instance)
                continue
                
            unscheduled_tasks.append(task)
            
        self.scaler(unscheduled_tasks)

        

    def scaler(self, unscheduled_tasks: List[pd.Series]):
        """
        Scale out by acquiring new instances for unscheduled tasks
        
        Args:
            unscheduled_tasks: List of tasks that couldn't be packed
        """
        # Group tasks by bin
        tasks_by_bin = {}
        for task in unscheduled_tasks:
            bin_idx = self._calculate_bin_index(task['runtime'])
            if bin_idx not in tasks_by_bin:
                tasks_by_bin[bin_idx] = []
            tasks_by_bin[bin_idx].append(task)
            
        # Process bins in descending order
        for bin_idx in sorted(tasks_by_bin.keys(), reverse=True):
            bin_tasks = tasks_by_bin[bin_idx]

            # Sort tasks by descending resource requirements (e.g., CPU or memory)
            bin_tasks.sort(key=lambda task: max(task['CPU_request'], task['memory_request']), reverse=True)
            
            while bin_tasks:
                # Create candidate groups
                best_score = -1
                best_instance_type = None
                best_group_size = 0
                
                for i in range(1, len(bin_tasks) + 1):
                    candidate_group = bin_tasks[:i]
                    
                    # Try each instance type
                    for _, instance_type in self.available_instance_types.iterrows():
                        if self._can_fit_group(candidate_group, instance_type):
                            score = self._calculate_score(candidate_group, instance_type)
                            if score > best_score:
                                best_score = score
                                best_instance_type = instance_type
                                best_group_size = i
                
                if best_instance_type is not None:
                    # Acquire new instance and assign tasks
                    instance = self._acquire_new_instance(best_instance_type, bin_idx)
                    for task in bin_tasks[:best_group_size]:
                        self._assign_task_to_instance(task, instance)
                    bin_tasks = bin_tasks[best_group_size:]
                else:
                    # Couldn't find suitable instance type
                    break
                    
    def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
        """Assign a task to an instance and update relevant metrics"""
        # Update instance resources

        # Ensure the instance exists
        # matching_indices = self.instance_bins.index[
        #     self.instance_bins['instance_ID'] == instance['instance_ID']
        # ]

        instance_idx = self.instance_bins.index[
            self.instance_bins['instance_ID'] == instance['instance_ID']
            ][0]
        
        self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
        self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

        # Update utilization metrics
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()

        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

        # try:
        #     if matching_indices.empty:
        #         raise ValueError(f"Instance ID {instance['instance_ID']} not found in instance_bins.")
        #     instance_idx = matching_indices[0]
        # except ValueError as e:
        #     print(f"Error: {e}")
        #     # Optionally log the error or handle it gracefully
        #     return  # Or take other appropriate action

        # Update instance details as required
        if self.instance_bins.at[instance_idx, 'runtime'] == 0: 
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
            self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
        else:
            max_timestamp = max(
                task['runtime'] + task['timestamp'], 
                self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
            )
            additional_runtime = task['runtime'] + task['timestamp'] - self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']
            if additional_runtime > 0:
                self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

        self.tasks += 1
        # Add task to task bins
        self.task_bins = pd.concat([
            self.task_bins,
            pd.DataFrame([{
                'job_ID': task['job_ID'],
                'task_index': task['task_index'],
                'bin_index': self._calculate_bin_index(task['runtime']),
                'instance_ID': instance['instance_ID'],
                'CPU_request': task['CPU_request'],
                'memory_request': task['memory_request'],
                'timestamp': task['timestamp'],
                'runtime': task['runtime']
            }])
        ], ignore_index=True)
        
    def _acquire_new_instance(self, instance_type: pd.Series, bin_idx: int) -> pd.Series:
        """Create a new instance of specified type and add to instance bins"""
        self.instance_counter += 1
        new_instance = pd.Series({
            'instance_ID': self.instance_counter,
            'bin_index': bin_idx,
            'CPU_capacity': instance_type['capacity_CPU'],
            'CPU_used': 0,
            'memory_capacity': instance_type['capacity_memory'],
            'memory_used': 0,
            'timestamp': 0,  # Will Set appropriate timestamp
            'runtime': 0,    # Will be set based on tasks
            'price' : instance_type['normalized_price']
        })

        # self.price_counter += instance_type['normalized_price']
        
        self.instance_bins = pd.concat([
            self.instance_bins, 
            pd.DataFrame([new_instance])
        ], ignore_index=True)

        # Update utilization metrics
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()

        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0
        
        return new_instance

    def _calculate_score(self, task_group: List[pd.Series], instance_type: pd.Series) -> float:
        """Calculate cost-efficiency score for a task group on an instance type"""
        # Calculate total resource usage
        total_cpu = sum(task['CPU_request'] for task in task_group)
        total_memory = sum(task['memory_request'] for task in task_group)
        
        # Determine constraining resource
        a_cpu_utilization = total_cpu / instance_type['capacity_CPU']
        a_memory_utilization = total_memory / instance_type['capacity_memory']
        
        constraining_resource = max(a_cpu_utilization, a_memory_utilization)
        
        # Calculate score
        return constraining_resource / instance_type['normalized_price']
        
    def _can_fit_group(self, task_group: List[pd.Series], instance_type: pd.Series) -> bool:
        """Check if a group of tasks can fit on an instance type"""
        total_cpu = sum(task['CPU_request'] for task in task_group)
        total_memory = sum(task['memory_request'] for task in task_group)
        
        return (total_cpu <= instance_type['capacity_CPU'] and 
                total_memory <= instance_type['capacity_memory'])
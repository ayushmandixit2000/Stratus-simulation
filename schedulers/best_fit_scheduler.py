import pandas as pd
import numpy as np

class BestFitScheduler:
    def __init__(self, available_instance_types: pd.DataFrame):
        """
        Initialize the Best-Fit scheduler

        Args:
            available_instance_types: DataFrame with columns ['capacity_CPU', 'capacity_memory', 'normalized_price']
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

        # change to active instances
        self.instance_counter = 0
        self.price_counter = 0
        self.tasks = 0
        self.cpu_utilization = 0.0  # Percentage of CPU resources used across all instances
        self.memory_utilization = 0.0  # Percentage of memory resources used across all instances

    def _calculate_bin_index(self, runtime: float) -> int:
        """Calculate bin index based on runtime using exponential binning"""
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1

    def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
        """Assign a task to an instance and update relevant metrics"""
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
            'timestamp': 0,  # Will set appropriate timestamp
            'runtime': 0,    # Will be set based on tasks
            'price': instance_type['normalized_price']
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

    def free_tasks_and_instances(self, current_timestamp):
        """Free expired tasks and instances at the given timestamp"""
        expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
        for _, task in expired_tasks.iterrows():
            instance_id = task['instance_ID']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

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

    def best_fit_scheduler(self, new_tasks: pd.DataFrame):
        """
        Best-Fit Scheduler: Assign tasks to the instance with closest remaining runtime.

        Args:
            new_tasks: DataFrame with task information
        """
        unscheduled_tasks = []

        for _, task in new_tasks.iterrows():
            task_assigned = False

            # Iterate through all existing instances
            # closest_instance = None
            # closest_runtime_diff = float('inf')

            best_instance = None
            min_remaining_resources = float('inf')

            for _, instance in self.instance_bins.iterrows():
                # Check if the instance can accommodate the task
                if (
                    instance['CPU_capacity'] - instance['CPU_used'] >= task['CPU_request'] and
                    instance['memory_capacity'] - instance['memory_used'] >= task['memory_request']
                ):
                    remaining_memory = instance['memory_capacity'] - instance['memory_used'] - task['memory_request']
                    remaining_cpu = instance['CPU_capacity'] - instance['CPU_used'] - task['CPU_request']
                    total_remaining_resources = remaining_cpu + remaining_memory

                    if total_remaining_resources < min_remaining_resources:
                        min_remaining_resources = total_remaining_resources
                        best_instance = instance

                    # remaining_runtime = instance['runtime'] - (
                    #     task['timestamp'] - instance['timestamp']
                    # )
                    # runtime_diff = abs(remaining_runtime - task['runtime'])

                    # if runtime_diff < closest_runtime_diff:
                    #     closest_runtime_diff = runtime_diff
                    #     closest_instance = instance

            # Assign to the closest instance if found
            if best_instance is not None:
                self._assign_task_to_instance(task, best_instance)
                task_assigned = True

            # If no existing instance can accommodate the task, acquire a new instance
            if not task_assigned:
                # Find the smallest instance type that can fit the task
                for _, instance_type in self.available_instance_types.iterrows():
                    if (
                        instance_type['capacity_CPU'] >= task['CPU_request'] and
                        instance_type['capacity_memory'] >= task['memory_request']
                    ):
                        # Acquire a new instance of this type
                        new_instance = self._acquire_new_instance(instance_type, self._calculate_bin_index(task['runtime']))
                        # Assign task to the new instance
                        self._assign_task_to_instance(task, new_instance)
                        task_assigned = True
                        break

            if not task_assigned:
                # Add to unscheduled tasks if no instance can be found
                unscheduled_tasks.append(task)

        return unscheduled_tasks
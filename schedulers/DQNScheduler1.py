import pandas as pd
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Experience Structure
# -------------------------
class Experience:
    def __init__(self, task_features, instance_features, placement_features,
                 bin_features, temporal_features, reward, next_state_features, done):
        self.task_features = task_features
        self.instance_features = instance_features
        self.placement_features = placement_features
        self.bin_features = bin_features
        self.temporal_features = temporal_features
        self.reward = reward
        self.next_state_features = next_state_features
        self.done = done

# -------------------------
# Basic Replay Buffer (Uniform Sampling)
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        return samples

# -------------------------
# Simple Scheduler Network (Basic DQN)
# -------------------------
class SimpleSchedulerNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleSchedulerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        # Scale output between 0 and 1
        q_value = torch.sigmoid(self.fc3(x))
        return q_value

# -------------------------
# Basic DNNScheduler: First Incremental RL Algorithm
# -------------------------
class DQNScheduler1:
    def __init__(self, available_instance_types: pd.DataFrame, task_dim: int, instance_dim: int, 
                 placement_dim: int, bin_dim: int, temporal_dim: int):
        """
        Initialize the DNNScheduler with basic RL.
        """
        self.available_instance_types = available_instance_types

        # DataFrames for tracking tasks and instances
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 'task_index', 'bin_index', 'instance_ID',
            'CPU_request', 'memory_request', 'timestamp', 'runtime'
        ])
        self.instance_bins = pd.DataFrame(columns=[
            'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
            'memory_capacity', 'memory_used', 'timestamp', 'runtime',
            'price', 'instance_type'
        ])

        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.price_counter = 0.0
        self.instance_counters = [0] * 10

        # Overbooking factor (kept same as final algorithm)
        self.overbooking_factor = 1.0

        # Hyperparameters for reward calculation (maintaining cost penalty logic)
        self.alpha = 1.0
        self.beta = 3.0  # cost penalty multiplier

        # Neural network and training parameters
        self.input_dim = task_dim + instance_dim + placement_dim + bin_dim + temporal_dim
        self.policy_network = SimpleSchedulerNetwork(self.input_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.batch_size = 32
        self.update_frequency = 10
        self.training_steps = 0

        # Basic replay buffer (uniform sampling)
        self.replay_buffer = ReplayBuffer(capacity=1000)

    # -------------------------
    # Feature Extraction Methods
    # -------------------------
    def _extract_task_features(self, task: pd.Series):
        return np.array([task['CPU_request'], task['memory_request'], task['runtime']])

    def _extract_instance_features(self, instance: pd.Series):
        return np.array([instance['CPU_capacity'], instance['memory_capacity'],
                         instance['CPU_used'], instance['memory_used']])

    def _extract_placement_features(self):
        return np.array([0])  # Placeholder for additional placement features

    def _extract_bin_features(self, bin_index: int):
        return np.array([bin_index])

    def _extract_temporal_features(self, task: pd.Series):
        return np.array([task['timestamp'], task['runtime']])

    def _calculate_bin_index(self, runtime: float) -> int:
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1

    # -------------------------
    # Scheduling and Assignment Methods
    # -------------------------
    def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
        instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
        self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
        self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()
        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

        # Price calculation logic (preserved from the final algorithm)
        if self.instance_bins.at[instance_idx, 'runtime'] == 0:
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
            self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
        else:
            prev_end = self.instance_bins.at[instance_idx, 'timestamp'] + self.instance_bins.at[instance_idx, 'runtime']
            new_end = task['timestamp'] + task['runtime']
            max_end = max(prev_end, new_end)
            additional_runtime = max_end - prev_end
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = max_end - task['timestamp']
            if additional_runtime > 0:
                self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

        self.tasks += 1
        new_task_entry = {
            'job_ID': task['job_ID'],
            'task_index': task['task_index'],
            'bin_index': self._calculate_bin_index(task['runtime']),
            'instance_ID': instance['instance_ID'],
            'CPU_request': task['CPU_request'],
            'memory_request': task['memory_request'],
            'timestamp': task['timestamp'],
            'runtime': task['runtime']
        }
        self.task_bins = pd.concat([self.task_bins, pd.DataFrame([new_task_entry])], ignore_index=True)

    def _acquire_new_instance(self, bin_idx: int) -> pd.Series:
        # Choose the cheapest available instance type
        cheapest = self.available_instance_types.loc[self.available_instance_types['normalized_price'].idxmin()]
        self.instance_counter += 1
        self.instance_id += 1
        instance_type_num = int(cheapest['IndexColumn']) - 1
        new_instance = pd.Series({
            'instance_ID': self.instance_id,
            'bin_index': bin_idx,
            'CPU_capacity': cheapest['capacity_CPU'] * self.overbooking_factor,
            'CPU_used': 0,
            'memory_capacity': cheapest['capacity_memory'] * self.overbooking_factor,
            'memory_used': 0,
            'timestamp': 0,
            'runtime': 0,
            'price': cheapest['normalized_price'],
            'instance_type': instance_type_num
        })

        self.instance_counters[instance_type_num] += 1

        self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
        return new_instance

    def _calculate_multi_objective_reward(self, task: pd.Series, instance: pd.Series) -> float:
        utilization_reward = (task['CPU_request'] + task['memory_request']) / (instance['CPU_capacity'] + instance['memory_capacity'])
        cost_penalty = instance['price']
        return utilization_reward - self.beta * cost_penalty

    # -------------------------
    # Neural Network Training (Basic DQN)
    # -------------------------
    def _train_network(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        experiences = self.replay_buffer.sample(self.batch_size)
        task_feats = torch.tensor(np.array([self._extract_task_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        inst_feats = torch.tensor(np.array([self._extract_instance_features(exp.instance_features) for exp in experiences]), dtype=torch.float32)
        place_feats = torch.tensor(np.array([self._extract_placement_features() for _ in experiences]), dtype=torch.float32)
        bin_feats = torch.tensor(np.array([self._extract_bin_features(exp.task_features.get('bin_index', 0)) for exp in experiences]), dtype=torch.float32)
        temp_feats = torch.tensor(np.array([self._extract_temporal_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        
        # Concatenate all features to form the input vector
        features = torch.cat([task_feats, inst_feats, place_feats, bin_feats, temp_feats], dim=1)
        q_values = self.policy_network(features)
        # Use immediate reward as the training target
        target_q_values = torch.tensor([[exp.reward] for exp in experiences], dtype=torch.float32)
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

    # -------------------------
    # Scheduling & Placement
    # -------------------------
    def schedule(self, new_tasks: pd.DataFrame):
        unscheduled_tasks = []
        sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
        
        for _, task in sorted_tasks.iterrows():
            bin_idx = self._calculate_bin_index(task['runtime'])
            candidate_instances = self.instance_bins[self.instance_bins['bin_index'] == bin_idx]
            candidate_instances = candidate_instances.sort_values('price')
            assigned = False
            
            if not candidate_instances.empty:
                for _, instance in candidate_instances.iterrows():
                    if ((instance['CPU_capacity'] - instance['CPU_used'] >= task['CPU_request']) and 
                        (instance['memory_capacity'] - instance['memory_used'] >= task['memory_request'])):
                        reward = self._calculate_multi_objective_reward(task, instance)
                        # Prepare features for network decision
                        task_feat = torch.tensor(self._extract_task_features(task), dtype=torch.float32).unsqueeze(0)
                        inst_feat = torch.tensor(self._extract_instance_features(instance), dtype=torch.float32).unsqueeze(0)
                        place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
                        bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
                        temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
                        features = torch.cat([task_feat, inst_feat, place_feat, bin_feat, temp_feat], dim=1)
                        score = self.policy_network(features)
                        # Simple threshold decision (tunable later)
                        if score.item() > 0.2:
                            self._assign_task_to_instance(task, instance)
                            exp = Experience(task_features=task, instance_features=instance, 
                                             placement_features=place_feat, bin_features=bin_feat,
                                             temporal_features=temp_feat, reward=reward, 
                                             next_state_features=None, done=False)
                            self.replay_buffer.add(exp)
                            assigned = True
                            break
            if not assigned:
                new_instance = self._acquire_new_instance(bin_idx)
                self._assign_task_to_instance(task, new_instance)
                reward = self._calculate_multi_objective_reward(task, new_instance)
                place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
                bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
                temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
                exp = Experience(task_features=task, instance_features=new_instance, 
                                 placement_features=place_feat, bin_features=bin_feat,
                                 temporal_features=temp_feat, reward=reward, 
                                 next_state_features=None, done=False)
                self.replay_buffer.add(exp)
            
            if self.training_steps % self.update_frequency == 0:
                self._train_network()
        
        return unscheduled_tasks

    # -------------------------
    # Free Expired Tasks and Instances
    # -------------------------
    def free_tasks_and_instances(self, current_timestamp):
        expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
        for _, task in expired_tasks.iterrows():
            instance_id = task['instance_ID']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
        expired_instances = self.instance_bins[self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp]
        self.instance_counter -= len(expired_instances)

        for _, instance in expired_instances.iterrows():
            instance_type_index = int(instance['instance_type'])
            self.instance_counters[instance_type_index] -= 1

        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()
        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0


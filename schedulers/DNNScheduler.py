# import pandas as pd
# import numpy as np
# from collections import deque
# from typing import List
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # -------------------------
# # Data Structure for Experience
# # -------------------------
# class Experience:
#     def __init__(self, task_features, instance_features, placement_features,
#                  bin_features, temporal_features, reward, next_state_features, done):
#         self.task_features = task_features
#         self.instance_features = instance_features
#         self.placement_features = placement_features
#         self.bin_features = bin_features
#         self.temporal_features = temporal_features
#         self.reward = reward
#         self.next_state_features = next_state_features
#         self.done = done

# # -------------------------
# # Prioritized Replay Buffer
# # -------------------------
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity: int, cleanup_frequency: int):
#         self.capacity = capacity
#         self.cleanup_frequency = cleanup_frequency
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)
#         self.access_counts = deque(maxlen=capacity)
#         self.alpha = 0.6
#         self.beta = 0.4
#         self.beta_increment = 0.001
#         self.epsilon = 1e-6

#     def add(self, experience: Experience, error: float):
#         priority = (abs(error) + self.epsilon) ** self.alpha
#         self.buffer.append(experience)
#         self.priorities.append(priority)
#         self.access_counts.append(0)
#         if len(self.buffer) % self.cleanup_frequency == 0:
#             self.cleanup_old_experiences()

#     def cleanup_old_experiences(self):
#         # Remove experiences with low access counts (bottom 20%)
#         if len(self.buffer) < 5:
#             return
#         counts = np.array(self.access_counts)
#         threshold = np.percentile(counts, 20)
#         new_buffer = deque(maxlen=self.capacity)
#         new_priorities = deque(maxlen=self.capacity)
#         new_access_counts = deque(maxlen=self.capacity)
#         for exp, prio, count in zip(self.buffer, self.priorities, self.access_counts):
#             if count > threshold:
#                 new_buffer.append(exp)
#                 new_priorities.append(prio)
#                 new_access_counts.append(count)
#         self.buffer = new_buffer
#         self.priorities = new_priorities
#         self.access_counts = new_access_counts

#     def sample(self, batch_size: int):
#         self.beta = min(1.0, self.beta + self.beta_increment)
#         priorities = np.array(self.priorities)
#         probabilities = priorities / priorities.sum()
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#         samples = [self.buffer[idx] for idx in indices]
#         # Update access counts for sampled experiences
#         for idx in indices:
#             self.access_counts[idx] += 1
#         # Compute importance sampling weights (normalized)
#         weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()
#         return samples, weights

# # -------------------------
# # Multi-Objective Neural Network
# # -------------------------
# class MultiObjectiveSchedulerNetwork(nn.Module):
#     def __init__(self, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         super(MultiObjectiveSchedulerNetwork, self).__init__()
#         # Process task features
#         self.task_network = nn.Sequential(
#             nn.Linear(task_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process instance features
#         self.instance_network = nn.Sequential(
#             nn.Linear(instance_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process placement features
#         self.placement_network = nn.Sequential(
#             nn.Linear(placement_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process bin features
#         self.bin_network = nn.Sequential(
#             nn.Linear(bin_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process temporal features
#         self.temporal_network = nn.Sequential(
#             nn.Linear(temporal_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Attention module to combine feature representations
#         self.attention_module = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.Tanh(),
#             nn.Linear(64, 32 * 5),
#             nn.Softmax(dim=1)
#         )
#         # Shared network for further processing
#         self.shared_network = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Heads for each objective: utilization, cost, SLA
#         self.utilization_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.cost_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.sla_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, task_features, instance_features, placement_features, bin_features, temporal_features):
#         t_out = self.task_network(task_features)
#         i_out = self.instance_network(instance_features)
#         p_out = self.placement_network(placement_features)
#         b_out = self.bin_network(bin_features)
#         temp_out = self.temporal_network(temporal_features)
#         # Concatenate all features
#         combined_features = torch.cat([t_out, i_out, p_out, b_out, temp_out], dim=1)
#         attention_weights = self.attention_module(combined_features)
#         attended_features = combined_features * attention_weights
#         shared_out = self.shared_network(attended_features)
#         utilization_score = self.utilization_head(shared_out)
#         cost_score = self.cost_head(shared_out)
#         sla_score = self.sla_head(shared_out)
#         # Combined score is the average of individual scores
#         combined_score = (utilization_score + cost_score + sla_score) / 3.0
#         return combined_score, utilization_score, cost_score, sla_score

# # -------------------------
# # Enhanced DNN Scheduler with Cost Optimization
# # -------------------------
# class DNNScheduler:
#     def __init__(self, available_instance_types: pd.DataFrame, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         """
#         Initialize the Enhanced DNN Scheduler with cost optimization.
#         """
#         self.available_instance_types = available_instance_types

#         # Initialize task and instance bins similar to StratusScheduler
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price', 'instance_type'
#         ])

#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0
#         self.price_counter = 0.0

#         # Additional state variables
#         self.workload_history = []
#         self.overbooking_factor = 1.0

#         # Neural network components and training parameters
#         self.policy_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.target_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
#         self.gamma = 0.99  # Discount factor
#         self.tau = 0.01    # Soft update factor
#         self.batch_size = 32
#         self.update_frequency = 10
#         self.training_steps = 0

#         # Initialize replay buffer for training the DNN
#         self.replay_buffer = PrioritizedReplayBuffer(capacity=1000, cleanup_frequency=50)

#     # -------------------------
#     # Feature Extraction Methods
#     # -------------------------
#     def _extract_task_features(self, task: pd.Series):
#         return np.array([task['CPU_request'], task['memory_request'], task['runtime']])

#     def _extract_instance_features(self, instance: pd.Series):
#         return np.array([instance['CPU_capacity'], instance['memory_capacity'],
#                          instance['CPU_used'], instance['memory_used']])

#     def _extract_placement_features(self):
#         # Placeholder for additional placement features
#         return np.array([0])

#     def _extract_bin_features(self, bin_index: int):
#         return np.array([bin_index])

#     def _extract_temporal_features(self, task: pd.Series):
#         return np.array([task['timestamp'], task['runtime']])

#     def _calculate_bin_index(self, runtime: float) -> int:
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     # -------------------------
#     # Scheduling and Assignment
#     # -------------------------
#     def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
#         # Update instance resource usage
#         instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             prev_end = self.instance_bins.at[instance_idx, 'timestamp'] + self.instance_bins.at[instance_idx, 'runtime']
#             new_end = task['timestamp'] + task['runtime']
#             max_end = max(prev_end, new_end)
#             additional_runtime = max_end - prev_end
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_end - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         self.tasks += 1
#         self.task_bins = pd.concat([
#             self.task_bins,
#             pd.DataFrame([{
#                 'job_ID': task['job_ID'],
#                 'task_index': task['task_index'],
#                 'bin_index': self._calculate_bin_index(task['runtime']),
#                 'instance_ID': instance['instance_ID'],
#                 'CPU_request': task['CPU_request'],
#                 'memory_request': task['memory_request'],
#                 'timestamp': task['timestamp'],
#                 'runtime': task['runtime']
#             }])
#         ], ignore_index=True)

#     def _acquire_new_instance(self, instance_type: pd.Series, bin_idx: int) -> pd.Series:
#         self.instance_counter += 1
#         self.instance_id += 1
#         instance_type_num = int(instance_type['IndexColumn']) - 1
#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price'],
#             'instance_type': instance_type_num
#         })
#         self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
#         return new_instance

#     def _calculate_multi_objective_reward(self, task: pd.Series, instance: pd.Series) -> float:
#         # Reward combines utilization (favoring high resource use) and cost (favoring lower price)
#         utilization_reward = (task['CPU_request'] + task['memory_request']) / (instance['CPU_capacity'] + instance['memory_capacity'])
#         cost_penalty = instance['price']
#         return utilization_reward - cost_penalty

#     # -------------------------
#     # Neural Network Training
#     # -------------------------
#     def _train_multi_objective_network(self):
#         if len(self.replay_buffer.buffer) < self.batch_size:
#             return
        
#         experiences, is_weights = self.replay_buffer.sample(self.batch_size)
#         # Convert experience features into tensors
#         task_feats = torch.tensor(np.array([self._extract_task_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
#         inst_feats = torch.tensor(np.array([self._extract_instance_features(exp.instance_features) for exp in experiences]), dtype=torch.float32)
#         place_feats = torch.tensor(np.array([self._extract_placement_features() for _ in experiences]), dtype=torch.float32)
#         # Using bin index from task features as a placeholder
#         bin_feats = torch.tensor(np.array([self._extract_bin_features(exp.task_features.get('bin_index', 0)) for exp in experiences]), dtype=torch.float32)
#         temp_feats = torch.tensor(np.array([self._extract_temporal_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        
#         q_values, _, _, _ = self.policy_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
#         with torch.no_grad():
#             target_q_values, _, _, _ = self.target_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
        
#         loss = ((q_values - target_q_values) ** 2 * torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1)).mean()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         # Soft update target network
#         for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
#         self.training_steps += 1

#     # -------------------------
#     # Scheduling & Placement Method
#     # -------------------------
#     def schedule(self, new_tasks: pd.DataFrame) -> List[pd.Series]:
#         unscheduled_tasks = []
#         sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
#         for _, task in sorted_tasks.iterrows():
#             bin_idx = self._calculate_bin_index(task['runtime'])
#             candidate_instances = self.instance_bins[self.instance_bins['bin_index'] == bin_idx]
#             assigned = False
#             if not candidate_instances.empty:
#                 for _, instance in candidate_instances.iterrows():
#                     if (instance['CPU_capacity'] - instance['CPU_used'] >= task['CPU_request'] and
#                         instance['memory_capacity'] - instance['memory_used'] >= task['memory_request']):
#                         reward = self._calculate_multi_objective_reward(task, instance)
#                         # Prepare features for the neural network (reshaping as needed)
#                         task_feat = torch.tensor(self._extract_task_features(task), dtype=torch.float32).unsqueeze(0)
#                         inst_feat = torch.tensor(self._extract_instance_features(instance), dtype=torch.float32).unsqueeze(0)
#                         place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
#                         bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
#                         temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
#                         score, util_score, cost_score, sla_score = self.policy_network(task_feat, inst_feat, place_feat, bin_feat, temp_feat)
#                         if score.item() > 0.5:  # example threshold for placement decision
#                             self._assign_task_to_instance(task, instance)
#                             exp = Experience(task_features=task, instance_features=instance, placement_features=place_feat,
#                                              bin_features=bin_feat, temporal_features=temp_feat,
#                                              reward=reward, next_state_features=None, done=False)
#                             self.replay_buffer.add(exp, error=-reward)
#                             assigned = True
#                             break
#             if not assigned:
#                 # Acquire a new instance if placement fails
#                 instance_type = self.available_instance_types.iloc[0]
#                 new_instance = self._acquire_new_instance(instance_type, bin_idx)
#                 self._assign_task_to_instance(task, new_instance)
#                 reward = self._calculate_multi_objective_reward(task, new_instance)
#                 place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
#                 bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
#                 temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
#                 exp = Experience(task_features=task, instance_features=new_instance, placement_features=place_feat,
#                                  bin_features=bin_feat, temporal_features=temp_feat,
#                                  reward=reward, next_state_features=None, done=False)
#                 self.replay_buffer.add(exp, error=-reward)
#             if self.training_steps % self.update_frequency == 0:
#                 self._train_multi_objective_network()
#         return unscheduled_tasks

#     # -------------------------
#     # Instance and Task Freeing
#     # -------------------------
#     def free_tasks_and_instances(self, current_timestamp):
#         expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
#         expired_instances = self.instance_bins[self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

# import pandas as pd
# import numpy as np
# from collections import deque
# from typing import List
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # -------------------------
# # Data Structure for Experience
# # -------------------------
# class Experience:
#     def __init__(self, task_features, instance_features, placement_features,
#                  bin_features, temporal_features, reward, next_state_features, done):
#         self.task_features = task_features
#         self.instance_features = instance_features
#         self.placement_features = placement_features
#         self.bin_features = bin_features
#         self.temporal_features = temporal_features
#         self.reward = reward
#         self.next_state_features = next_state_features
#         self.done = done

# # -------------------------
# # Prioritized Replay Buffer
# # -------------------------
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity: int, cleanup_frequency: int):
#         self.capacity = capacity
#         self.cleanup_frequency = cleanup_frequency
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)
#         self.access_counts = deque(maxlen=capacity)
#         self.alpha = 0.6
#         self.beta = 0.4
#         self.beta_increment = 0.001
#         self.epsilon = 1e-6

#     def add(self, experience: Experience, error: float):
#         priority = (abs(error) + self.epsilon) ** self.alpha
#         self.buffer.append(experience)
#         self.priorities.append(priority)
#         self.access_counts.append(0)
#         if len(self.buffer) % self.cleanup_frequency == 0:
#             self.cleanup_old_experiences()

#     def cleanup_old_experiences(self):
#         if len(self.buffer) < 5:
#             return
#         counts = np.array(self.access_counts)
#         threshold = np.percentile(counts, 20)
#         new_buffer = deque(maxlen=self.capacity)
#         new_priorities = deque(maxlen=self.capacity)
#         new_access_counts = deque(maxlen=self.capacity)
#         for exp, prio, count in zip(self.buffer, self.priorities, self.access_counts):
#             if count > threshold:
#                 new_buffer.append(exp)
#                 new_priorities.append(prio)
#                 new_access_counts.append(count)
#         self.buffer = new_buffer
#         self.priorities = new_priorities
#         self.access_counts = new_access_counts

#     def sample(self, batch_size: int):
#         self.beta = min(1.0, self.beta + self.beta_increment)
#         priorities = np.array(self.priorities)
#         probabilities = priorities / priorities.sum()
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#         samples = [self.buffer[idx] for idx in indices]
#         for idx in indices:
#             self.access_counts[idx] += 1
#         weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()
#         return samples, weights

# # -------------------------
# # Multi-Objective Neural Network
# # -------------------------
# class MultiObjectiveSchedulerNetwork(nn.Module):
#     def __init__(self, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         super(MultiObjectiveSchedulerNetwork, self).__init__()
#         # Process task features
#         self.task_network = nn.Sequential(
#             nn.Linear(task_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process instance features
#         self.instance_network = nn.Sequential(
#             nn.Linear(instance_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process placement features
#         self.placement_network = nn.Sequential(
#             nn.Linear(placement_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process bin features
#         self.bin_network = nn.Sequential(
#             nn.Linear(bin_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process temporal features
#         self.temporal_network = nn.Sequential(
#             nn.Linear(temporal_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Attention module to combine feature representations
#         self.attention_module = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.Tanh(),
#             nn.Linear(64, 32 * 5),
#             nn.Softmax(dim=1)
#         )
#         # Shared network for further processing
#         self.shared_network = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Heads for each objective: utilization, cost, SLA
#         self.utilization_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.cost_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.sla_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, task_features, instance_features, placement_features, bin_features, temporal_features):
#         t_out = self.task_network(task_features)
#         i_out = self.instance_network(instance_features)
#         p_out = self.placement_network(placement_features)
#         b_out = self.bin_network(bin_features)
#         temp_out = self.temporal_network(temporal_features)
#         combined_features = torch.cat([t_out, i_out, p_out, b_out, temp_out], dim=1)
#         attention_weights = self.attention_module(combined_features)
#         attended_features = combined_features * attention_weights
#         shared_out = self.shared_network(attended_features)
#         utilization_score = self.utilization_head(shared_out)
#         cost_score = self.cost_head(shared_out)
#         sla_score = self.sla_head(shared_out)
#         combined_score = (utilization_score + cost_score + sla_score) / 3.0
#         return combined_score, utilization_score, cost_score, sla_score

# # -------------------------
# # Enhanced DNN Scheduler with Aggressive Cost Optimization
# # -------------------------
# class DNNScheduler:
#     def __init__(self, available_instance_types: pd.DataFrame, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         """
#         Initialize the DNNScheduler with aggressive cost optimization.
#         """
#         self.available_instance_types = available_instance_types

#         # DataFrames for task and instance bins
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price', 'instance_type'
#         ])

#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0
#         self.price_counter = 0.0

#         # Overbooking factor: set conservatively to avoid over-provisioning
#         self.overbooking_factor = 1.0

#         # Hyperparameters to favor cost savings (increase cost penalty)
#         self.alpha = 1.0  # utilization reward weight
#         self.beta = 2.5   # cost penalty weight

#         # Neural network and RL training parameters
#         self.policy_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.target_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
#         self.gamma = 0.99
#         self.tau = 0.01
#         self.batch_size = 32
#         self.update_frequency = 10
#         self.training_steps = 0

#         self.replay_buffer = PrioritizedReplayBuffer(capacity=1000, cleanup_frequency=50)

#     # -------------------------
#     # Feature Extraction Methods
#     # -------------------------
#     def _extract_task_features(self, task: pd.Series):
#         return np.array([task['CPU_request'], task['memory_request'], task['runtime']])

#     def _extract_instance_features(self, instance: pd.Series):
#         return np.array([instance['CPU_capacity'], instance['memory_capacity'],
#                          instance['CPU_used'], instance['memory_used']])

#     def _extract_placement_features(self):
#         return np.array([0])  # Placeholder

#     def _extract_bin_features(self, bin_index: int):
#         return np.array([bin_index])

#     def _extract_temporal_features(self, task: pd.Series):
#         return np.array([task['timestamp'], task['runtime']])

#     def _calculate_bin_index(self, runtime: float) -> int:
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     # -------------------------
#     # Scheduling and Assignment
#     # -------------------------
#     def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
#         instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             prev_end = self.instance_bins.at[instance_idx, 'timestamp'] + self.instance_bins.at[instance_idx, 'runtime']
#             new_end = task['timestamp'] + task['runtime']
#             max_end = max(prev_end, new_end)
#             additional_runtime = max_end - prev_end
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_end - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         self.tasks += 1
#         new_task_entry = {
#             'job_ID': task['job_ID'],
#             'task_index': task['task_index'],
#             'bin_index': self._calculate_bin_index(task['runtime']),
#             'instance_ID': instance['instance_ID'],
#             'CPU_request': task['CPU_request'],
#             'memory_request': task['memory_request'],
#             'timestamp': task['timestamp'],
#             'runtime': task['runtime']
#         }
#         self.task_bins = pd.concat([self.task_bins, pd.DataFrame([new_task_entry])], ignore_index=True)

#     def _acquire_new_instance(self, bin_idx: int) -> pd.Series:
#         # Instead of defaulting to the first type, choose the cheapest instance type
#         cheapest = self.available_instance_types.loc[self.available_instance_types['normalized_price'].idxmin()]
#         self.instance_counter += 1
#         self.instance_id += 1
#         instance_type_num = int(cheapest['IndexColumn']) - 1
#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': cheapest['capacity_CPU'] * self.overbooking_factor,
#             'CPU_used': 0,
#             'memory_capacity': cheapest['capacity_memory'] * self.overbooking_factor,
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': cheapest['normalized_price'],
#             'instance_type': instance_type_num
#         })
#         self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
#         return new_instance

#     def _calculate_multi_objective_reward(self, task: pd.Series, instance: pd.Series) -> float:
#         utilization_reward = (task['CPU_request'] + task['memory_request']) / (instance['CPU_capacity'] + instance['memory_capacity'])
#         cost_penalty = instance['price']
#         return utilization_reward - self.beta * cost_penalty

#     # -------------------------
#     # Neural Network Training
#     # -------------------------
#     def _train_multi_objective_network(self):
#         if len(self.replay_buffer.buffer) < self.batch_size:
#             return
#         experiences, is_weights = self.replay_buffer.sample(self.batch_size)
#         task_feats = torch.tensor(np.array([self._extract_task_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
#         inst_feats = torch.tensor(np.array([self._extract_instance_features(exp.instance_features) for exp in experiences]), dtype=torch.float32)
#         place_feats = torch.tensor(np.array([self._extract_placement_features() for _ in experiences]), dtype=torch.float32)
#         bin_feats = torch.tensor(np.array([self._extract_bin_features(exp.task_features.get('bin_index', 0)) for exp in experiences]), dtype=torch.float32)
#         temp_feats = torch.tensor(np.array([self._extract_temporal_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        
#         q_values, _, _, _ = self.policy_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
#         with torch.no_grad():
#             target_q_values, _, _, _ = self.target_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
        
#         loss = ((q_values - target_q_values) ** 2 * torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1)).mean()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
#         self.training_steps += 1

#     # -------------------------
#     # Simple Right-Sizing to Terminate Idle Instances
#     # -------------------------
#     def _rightsize_instances(self, current_timestamp):
#         idle_threshold = 0.1  # 10% utilization threshold
#         removable = []
#         for idx, instance in self.instance_bins.iterrows():
#             if instance['CPU_capacity'] > 0:
#                 utilization = instance['CPU_used'] / instance['CPU_capacity']
#             else:
#                 utilization = 0
#             # If utilization is very low and the instance has been running longer than its runtime, mark it
#             if utilization < idle_threshold and (current_timestamp - instance['timestamp'] > instance['runtime']):
#                 removable.append(idx)
#         if removable:
#             self.instance_bins = self.instance_bins.drop(removable)
#             self.instance_counter -= len(removable)

#     # -------------------------
#     # Scheduling & Placement Method
#     # -------------------------
#     def schedule(self, new_tasks: pd.DataFrame) -> List[pd.Series]:
#         unscheduled_tasks = []
#         sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
        
#         # For each task, try to place it on a suitable instance
#         for _, task in sorted_tasks.iterrows():
#             bin_idx = self._calculate_bin_index(task['runtime'])
#             # Consider candidate instances in the same bin; sort by price (cheaper first)
#             candidate_instances = self.instance_bins[self.instance_bins['bin_index'] == bin_idx]
#             candidate_instances = candidate_instances.sort_values('price')
#             assigned = False
#             if not candidate_instances.empty:
#                 for _, instance in candidate_instances.iterrows():
#                     if ((instance['CPU_capacity'] - instance['CPU_used'] >= task['CPU_request']) and 
#                         (instance['memory_capacity'] - instance['memory_used'] >= task['memory_request'])):
#                         reward = self._calculate_multi_objective_reward(task, instance)
#                         # Prepare features
#                         task_feat = torch.tensor(self._extract_task_features(task), dtype=torch.float32).unsqueeze(0)
#                         inst_feat = torch.tensor(self._extract_instance_features(instance), dtype=torch.float32).unsqueeze(0)
#                         place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
#                         bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
#                         temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
#                         # Lower threshold (0.3) to favor existing instance placement over acquiring new ones
#                         score, _, _, _ = self.policy_network(task_feat, inst_feat, place_feat, bin_feat, temp_feat)
#                         if score.item() > 0.3:
#                             self._assign_task_to_instance(task, instance)
#                             exp = Experience(task_features=task, instance_features=instance, placement_features=place_feat,
#                                              bin_features=bin_feat, temporal_features=temp_feat,
#                                              reward=reward, next_state_features=None, done=False)
#                             self.replay_buffer.add(exp, error=-reward)
#                             assigned = True
#                             break
#             if not assigned:
#                 # Acquire a new instance from the cheapest available type
#                 new_instance = self._acquire_new_instance(bin_idx)
#                 self._assign_task_to_instance(task, new_instance)
#                 reward = self._calculate_multi_objective_reward(task, new_instance)
#                 place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
#                 bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
#                 temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
#                 exp = Experience(task_features=task, instance_features=new_instance, placement_features=place_feat,
#                                  bin_features=bin_feat, temporal_features=temp_feat,
#                                  reward=reward, next_state_features=None, done=False)
#                 self.replay_buffer.add(exp, error=-reward)
#             if self.training_steps % self.update_frequency == 0:
#                 self._train_multi_objective_network()
        
#         # Optionally right-size instances after scheduling to shut down idle, expensive resources
#         if not new_tasks.empty:
#             current_timestamp = new_tasks.iloc[0]['timestamp']
#             self._rightsize_instances(current_timestamp)
#         return unscheduled_tasks

#     # -------------------------
#     # Free Expired Tasks and Instances
#     # -------------------------
#     def free_tasks_and_instances(self, current_timestamp):
#         expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
#         expired_instances = self.instance_bins[self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

import pandas as pd
import numpy as np
from collections import deque
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Data Structure for Experience
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
# Prioritized Replay Buffer
# -------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, cleanup_frequency: int):
        self.capacity = capacity
        self.cleanup_frequency = cleanup_frequency
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.access_counts = deque(maxlen=capacity)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon = 1e-6

    def add(self, experience: Experience, error: float):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.access_counts.append(0)
        if len(self.buffer) % self.cleanup_frequency == 0:
            self.cleanup_old_experiences()

    def cleanup_old_experiences(self):
        if len(self.buffer) < 5:
            return
        counts = np.array(self.access_counts)
        threshold = np.percentile(counts, 20)
        new_buffer = deque(maxlen=self.capacity)
        new_priorities = deque(maxlen=self.capacity)
        new_access_counts = deque(maxlen=self.capacity)
        for exp, prio, count in zip(self.buffer, self.priorities, self.access_counts):
            if count > threshold:
                new_buffer.append(exp)
                new_priorities.append(prio)
                new_access_counts.append(count)
        self.buffer = new_buffer
        self.priorities = new_priorities
        self.access_counts = new_access_counts

    def sample(self, batch_size: int):
        self.beta = min(1.0, self.beta + self.beta_increment)
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        for idx in indices:
            self.access_counts[idx] += 1
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, weights

# -------------------------
# Multi-Objective Neural Network
# -------------------------
class MultiObjectiveSchedulerNetwork(nn.Module):
    def __init__(self, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
        super(MultiObjectiveSchedulerNetwork, self).__init__()
        # Process task features
        self.task_network = nn.Sequential(
            nn.Linear(task_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Process instance features
        self.instance_network = nn.Sequential(
            nn.Linear(instance_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Process placement features
        self.placement_network = nn.Sequential(
            nn.Linear(placement_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Process bin features
        self.bin_network = nn.Sequential(
            nn.Linear(bin_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Process temporal features
        self.temporal_network = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Attention module to combine feature representations
        self.attention_module = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.Tanh(),
            nn.Linear(64, 32 * 5),
            nn.Softmax(dim=1)
        )
        # Shared network for further processing
        self.shared_network = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Heads for each objective: utilization, cost, SLA
        self.utilization_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.cost_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.sla_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, task_features, instance_features, placement_features, bin_features, temporal_features):
        t_out = self.task_network(task_features)
        i_out = self.instance_network(instance_features)
        p_out = self.placement_network(placement_features)
        b_out = self.bin_network(bin_features)
        temp_out = self.temporal_network(temporal_features)
        combined_features = torch.cat([t_out, i_out, p_out, b_out, temp_out], dim=1)
        attention_weights = self.attention_module(combined_features)
        attended_features = combined_features * attention_weights
        shared_out = self.shared_network(attended_features)
        utilization_score = self.utilization_head(shared_out)
        cost_score = self.cost_head(shared_out)
        sla_score = self.sla_head(shared_out)
        combined_score = (utilization_score + cost_score + sla_score) / 3.0
        return combined_score, utilization_score, cost_score, sla_score

# -------------------------
# Enhanced DNN Scheduler with Advanced Cost Optimization
# -------------------------
class DNNScheduler:
    def __init__(self, available_instance_types: pd.DataFrame, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
        """
        Initialize the DNNScheduler with advanced cost optimization.
        """
        self.available_instance_types = available_instance_types

        # DataFrames for tasks and instance bins
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

        # Overbooking factor adjusted conservatively based on cost history
        self.overbooking_factor = 1.0

        # Hyperparameters for reward: aggressively penalize cost
        self.alpha = 1.0
        self.beta = 3.0  # increased cost penalty

        # Migration threshold: if an instance's price is significantly above the cheapest,
        # attempt to migrate its tasks.
        self.migration_threshold = 1.5  # 50% more expensive than cheapest

        # Neural network and training parameters
        self.policy_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
        self.target_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 32
        self.update_frequency = 10
        self.training_steps = 0

        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000, cleanup_frequency=50)

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
    # Scheduling and Assignment
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
        # Choose the cheapest instance type available
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
        self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
        return new_instance

    def _calculate_multi_objective_reward(self, task: pd.Series, instance: pd.Series) -> float:
        utilization_reward = (task['CPU_request'] + task['memory_request']) / (instance['CPU_capacity'] + instance['memory_capacity'])
        cost_penalty = instance['price']
        return utilization_reward - self.beta * cost_penalty

    # -------------------------
    # Migration: Consolidate Tasks from Expensive Instances
    # -------------------------
    def _migrate_tasks(self):
        # Identify the cheapest instance price among active instances
        if self.instance_bins.empty:
            return
        min_price = self.instance_bins['price'].min()
        expensive_instances = self.instance_bins[self.instance_bins['price'] > self.migration_threshold * min_price]

        for idx, expensive in expensive_instances.iterrows():
            # Find tasks on the expensive instance
            tasks_on_instance = self.task_bins[self.task_bins['instance_ID'] == expensive['instance_ID']]
            for _, task in tasks_on_instance.iterrows():
                bin_idx = task['bin_index']
                # Look for a cheaper candidate instance in the same bin
                candidates = self.instance_bins[(self.instance_bins['bin_index'] == bin_idx) &
                                                (self.instance_bins['price'] < expensive['price'])]
                candidates = candidates.sort_values('price')
                migrated = False
                for _, candidate in candidates.iterrows():
                    if ((candidate['CPU_capacity'] - candidate['CPU_used'] >= task['CPU_request']) and
                        (candidate['memory_capacity'] - candidate['memory_used'] >= task['memory_request'])):
                        # Migrate task: update assignment in task_bins and adjust candidate's resources
                        self._assign_task_to_instance(task, candidate)
                        # Remove task from the expensive instance's record
                        self.task_bins = self.task_bins[self.task_bins['instance_ID'] != expensive['instance_ID']]
                        migrated = True
                        break
                if migrated:
                    # Optionally log migration event
                    pass

    # -------------------------
    # Neural Network Training
    # -------------------------
    def _train_multi_objective_network(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        experiences, is_weights = self.replay_buffer.sample(self.batch_size)
        task_feats = torch.tensor(np.array([self._extract_task_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        inst_feats = torch.tensor(np.array([self._extract_instance_features(exp.instance_features) for exp in experiences]), dtype=torch.float32)
        place_feats = torch.tensor(np.array([self._extract_placement_features() for _ in experiences]), dtype=torch.float32)
        bin_feats = torch.tensor(np.array([self._extract_bin_features(exp.task_features.get('bin_index', 0)) for exp in experiences]), dtype=torch.float32)
        temp_feats = torch.tensor(np.array([self._extract_temporal_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        
        q_values, _, _, _ = self.policy_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
        with torch.no_grad():
            target_q_values, _, _, _ = self.target_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
        
        loss = ((q_values - target_q_values) ** 2 * torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        self.training_steps += 1

    # -------------------------
    # Enhanced Right-Sizing to Terminate Idle Instances
    # -------------------------
    def _rightsize_instances(self, current_timestamp):
        idle_threshold = 0.05  # 5% utilization threshold for termination
        removable = []
        for idx, instance in self.instance_bins.iterrows():
            if instance['CPU_capacity'] > 0:
                utilization = instance['CPU_used'] / instance['CPU_capacity']
            else:
                utilization = 0
            if utilization < idle_threshold and (current_timestamp - instance['timestamp'] > instance['runtime']):
                removable.append(idx)
        if removable:
            self.instance_bins = self.instance_bins.drop(removable)
            self.instance_counter -= len(removable)

    # -------------------------
    # Scheduling & Placement Method
    # -------------------------
    def schedule(self, new_tasks: pd.DataFrame) -> List[pd.Series]:
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
                        # Prepare features for NN decision
                        task_feat = torch.tensor(self._extract_task_features(task), dtype=torch.float32).unsqueeze(0)
                        inst_feat = torch.tensor(self._extract_instance_features(instance), dtype=torch.float32).unsqueeze(0)
                        place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
                        bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
                        temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
                        # Lower threshold further (0.2) to favor reusing cheap instances
                        score, _, _, _ = self.policy_network(task_feat, inst_feat, place_feat, bin_feat, temp_feat)
                        if score.item() > 0.2:
                            self._assign_task_to_instance(task, instance)
                            exp = Experience(task_features=task, instance_features=instance, placement_features=place_feat,
                                             bin_features=bin_feat, temporal_features=temp_feat,
                                             reward=reward, next_state_features=None, done=False)
                            self.replay_buffer.add(exp, error=-reward)
                            assigned = True
                            break
            if not assigned:
                new_instance = self._acquire_new_instance(bin_idx)
                self._assign_task_to_instance(task, new_instance)
                reward = self._calculate_multi_objective_reward(task, new_instance)
                place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
                bin_feat = torch.tensor(self._extract_bin_features(bin_idx), dtype=torch.float32).unsqueeze(0)
                temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
                exp = Experience(task_features=task, instance_features=new_instance, placement_features=place_feat,
                                 bin_features=bin_feat, temporal_features=temp_feat,
                                 reward=reward, next_state_features=None, done=False)
                self.replay_buffer.add(exp, error=-reward)
            if self.training_steps % self.update_frequency == 0:
                self._train_multi_objective_network()
        
        # After scheduling, attempt migration to consolidate tasks from expensive instances
        self._migrate_tasks()
        
        # Right-size idle instances
        if not new_tasks.empty:
            current_timestamp = new_tasks.iloc[0]['timestamp']
            self._rightsize_instances(current_timestamp)
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
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
        total_memory_capacity = self.instance_bins['memory_capacity'].sum()
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()
        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

# import pandas as pd
# import numpy as np
# from collections import deque
# from typing import List
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # -------------------------
# # Data Structure for Experience
# # -------------------------
# class Experience:
#     def __init__(self, task_features, instance_features, placement_features,
#                  bin_features, temporal_features, reward, next_state_features, done):
#         self.task_features = task_features
#         self.instance_features = instance_features
#         self.placement_features = placement_features
#         self.bin_features = bin_features
#         self.temporal_features = temporal_features
#         self.reward = reward
#         self.next_state_features = next_state_features
#         self.done = done

# # -------------------------
# # Prioritized Replay Buffer
# # -------------------------
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity: int, cleanup_frequency: int):
#         self.capacity = capacity
#         self.cleanup_frequency = cleanup_frequency
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)
#         self.access_counts = deque(maxlen=capacity)
#         self.alpha = 0.6
#         self.beta = 0.4
#         self.beta_increment = 0.001
#         self.epsilon = 1e-6

#     def add(self, experience: Experience, error: float):
#         priority = (abs(error) + self.epsilon) ** self.alpha
#         self.buffer.append(experience)
#         self.priorities.append(priority)
#         self.access_counts.append(0)
#         if len(self.buffer) % self.cleanup_frequency == 0:
#             self.cleanup_old_experiences()

#     def cleanup_old_experiences(self):
#         if len(self.buffer) < 5:
#             return
#         counts = np.array(self.access_counts)
#         threshold = np.percentile(counts, 20)
#         new_buffer = deque(maxlen=self.capacity)
#         new_priorities = deque(maxlen=self.capacity)
#         new_access_counts = deque(maxlen=self.capacity)
#         for exp, prio, count in zip(self.buffer, self.priorities, self.access_counts):
#             if count > threshold:
#                 new_buffer.append(exp)
#                 new_priorities.append(prio)
#                 new_access_counts.append(count)
#         self.buffer = new_buffer
#         self.priorities = new_priorities
#         self.access_counts = new_access_counts

#     def sample(self, batch_size: int):
#         self.beta = min(1.0, self.beta + self.beta_increment)
#         priorities = np.array(self.priorities)
#         probabilities = priorities / priorities.sum()
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#         samples = [self.buffer[idx] for idx in indices]
#         for idx in indices:
#             self.access_counts[idx] += 1
#         weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()
#         return samples, weights

# # -------------------------
# # Multi-Objective Neural Network
# # -------------------------
# class MultiObjectiveSchedulerNetwork(nn.Module):
#     def __init__(self, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         super(MultiObjectiveSchedulerNetwork, self).__init__()
#         # Process task features
#         self.task_network = nn.Sequential(
#             nn.Linear(task_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process instance features
#         self.instance_network = nn.Sequential(
#             nn.Linear(instance_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process placement features
#         self.placement_network = nn.Sequential(
#             nn.Linear(placement_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process bin features
#         self.bin_network = nn.Sequential(
#             nn.Linear(bin_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Process temporal features
#         self.temporal_network = nn.Sequential(
#             nn.Linear(temporal_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Attention module to combine feature representations
#         self.attention_module = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.Tanh(),
#             nn.Linear(64, 32 * 5),
#             nn.Softmax(dim=1)
#         )
#         # Shared network for further processing
#         self.shared_network = nn.Sequential(
#             nn.Linear(32 * 5, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         # Heads for each objective: utilization, cost, SLA
#         self.utilization_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.cost_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )
#         self.sla_head = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, task_features, instance_features, placement_features, bin_features, temporal_features):
#         t_out = self.task_network(task_features)
#         i_out = self.instance_network(instance_features)
#         p_out = self.placement_network(placement_features)
#         b_out = self.bin_network(bin_features)
#         temp_out = self.temporal_network(temporal_features)
#         combined_features = torch.cat([t_out, i_out, p_out, b_out, temp_out], dim=1)
#         attention_weights = self.attention_module(combined_features)
#         attended_features = combined_features * attention_weights
#         shared_out = self.shared_network(attended_features)
#         utilization_score = self.utilization_head(shared_out)
#         cost_score = self.cost_head(shared_out)
#         sla_score = self.sla_head(shared_out)
#         combined_score = (utilization_score + cost_score + sla_score) / 3.0
#         return combined_score, utilization_score, cost_score, sla_score

# # -------------------------
# # Enhanced DNNScheduler with Advanced Cost Optimization & Packing
# # -------------------------
# class DNNScheduler:
#     def __init__(self, available_instance_types: pd.DataFrame, task_dim, instance_dim, placement_dim, bin_dim, temporal_dim):
#         """
#         Initialize the DNNScheduler with advanced cost optimization.
#         """
#         self.available_instance_types = available_instance_types

#         # DataFrames for tasks and instance bins
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price', 'instance_type'
#         ])

#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0
#         self.price_counter = 0.0

#         # Overbooking factor (kept conservative)
#         self.overbooking_factor = 1.0

#         # Hyperparameters for cost: aggressively penalize cost
#         self.alpha = 1.0
#         self.beta = 3.0  # increased cost penalty

#         # Migration threshold: migrate tasks from instances 50% more expensive than cheapest
#         self.migration_threshold = 1.5

#         # Neural network and RL training parameters
#         self.policy_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.target_network = MultiObjectiveSchedulerNetwork(task_dim, instance_dim, placement_dim, bin_dim, temporal_dim)
#         self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
#         self.gamma = 0.99
#         self.tau = 0.01
#         self.batch_size = 32
#         self.update_frequency = 10
#         self.training_steps = 0

#         self.replay_buffer = PrioritizedReplayBuffer(capacity=1000, cleanup_frequency=50)

#     # -------------------------
#     # Feature Extraction Methods
#     # -------------------------
#     def _extract_task_features(self, task: pd.Series):
#         return np.array([task['CPU_request'], task['memory_request'], task['runtime']])

#     def _extract_instance_features(self, instance: pd.Series):
#         return np.array([instance['CPU_capacity'], instance['memory_capacity'],
#                          instance['CPU_used'], instance['memory_used']])

#     def _extract_placement_features(self):
#         return np.array([0])  # Placeholder

#     def _extract_bin_features(self, bin_index: int):
#         return np.array([bin_index])

#     def _extract_temporal_features(self, task: pd.Series):
#         return np.array([task['timestamp'], task['runtime']])

#     def _calculate_bin_index(self, runtime: float) -> int:
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     # -------------------------
#     # Scheduling & Assignment
#     # -------------------------
#     def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
#         instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             prev_end = self.instance_bins.at[instance_idx, 'timestamp'] + self.instance_bins.at[instance_idx, 'runtime']
#             new_end = task['timestamp'] + task['runtime']
#             max_end = max(prev_end, new_end)
#             additional_runtime = max_end - prev_end
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_end - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         self.tasks += 1
#         new_task_entry = {
#             'job_ID': task['job_ID'],
#             'task_index': task['task_index'],
#             'bin_index': self._calculate_bin_index(task['runtime']),
#             'instance_ID': instance['instance_ID'],
#             'CPU_request': task['CPU_request'],
#             'memory_request': task['memory_request'],
#             'timestamp': task['timestamp'],
#             'runtime': task['runtime']
#         }
#         self.task_bins = pd.concat([self.task_bins, pd.DataFrame([new_task_entry])], ignore_index=True)

#     def _acquire_new_instance(self, bin_idx: int) -> pd.Series:
#         # Always choose the cheapest available instance type
#         cheapest = self.available_instance_types.loc[self.available_instance_types['normalized_price'].idxmin()]
#         self.instance_counter += 1
#         self.instance_id += 1
#         instance_type_num = int(cheapest['IndexColumn']) - 1
#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': cheapest['capacity_CPU'] * self.overbooking_factor,
#             'CPU_used': 0,
#             'memory_capacity': cheapest['capacity_memory'] * self.overbooking_factor,
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': cheapest['normalized_price'],
#             'instance_type': instance_type_num
#         })
#         self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
#         return new_instance

#     def _calculate_multi_objective_reward(self, task: pd.Series, instance: pd.Series) -> float:
#         utilization_reward = (task['CPU_request'] + task['memory_request']) / (instance['CPU_capacity'] + instance['memory_capacity'])
#         cost_penalty = instance['price']
#         return utilization_reward - self.beta * cost_penalty

#     # -------------------------
#     # Group-Based Packing (Packer)
#     # -------------------------
#     def packer(self, new_tasks: pd.DataFrame) -> List[pd.Series]:
#         unscheduled_tasks = []
#         sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
#         for _, task in sorted_tasks.iterrows():
#             task_bin = self._calculate_bin_index(task['runtime'])
#             # Up-packing: try same bin first
#             same_bin_instances = self.instance_bins[self.instance_bins['bin_index'] == task_bin]
#             eligible = same_bin_instances[
#                 (same_bin_instances['CPU_capacity'] - same_bin_instances['CPU_used'] >= task['CPU_request']) &
#                 (same_bin_instances['memory_capacity'] - same_bin_instances['memory_used'] >= task['memory_request'])
#             ]
#             if not eligible.empty:
#                 candidate = eligible.sort_values('price').iloc[0]
#                 task_feat = torch.tensor(self._extract_task_features(task), dtype=torch.float32).unsqueeze(0)
#                 inst_feat = torch.tensor(self._extract_instance_features(candidate), dtype=torch.float32).unsqueeze(0)
#                 place_feat = torch.tensor(self._extract_placement_features(), dtype=torch.float32).unsqueeze(0)
#                 bin_feat = torch.tensor(self._extract_bin_features(task_bin), dtype=torch.float32).unsqueeze(0)
#                 temp_feat = torch.tensor(self._extract_temporal_features(task), dtype=torch.float32).unsqueeze(0)
#                 score, _, _, _ = self.policy_network(task_feat, inst_feat, place_feat, bin_feat, temp_feat)
#                 if score.item() > 0.2:
#                     self._assign_task_to_instance(task, candidate)
#                     continue
#             # Higher bin: if same bin not available, try instances with a larger bin index
#             higher_bin_instances = self.instance_bins[self.instance_bins['bin_index'] > task_bin]
#             eligible = higher_bin_instances[
#                 (higher_bin_instances['CPU_capacity'] - higher_bin_instances['CPU_used'] >= task['CPU_request']) &
#                 (higher_bin_instances['memory_capacity'] - higher_bin_instances['memory_used'] >= task['memory_request'])
#             ]
#             if not eligible.empty:
#                 candidate = eligible.sort_values('price').iloc[0]
#                 self._assign_task_to_instance(task, candidate)
#                 continue
#             # Down-packing: try lower bin instances by promoting them
#             lower_bin_instances = self.instance_bins[self.instance_bins['bin_index'] < task_bin]
#             eligible = lower_bin_instances[
#                 (lower_bin_instances['CPU_capacity'] - lower_bin_instances['CPU_used'] >= task['CPU_request']) &
#                 (lower_bin_instances['memory_capacity'] - lower_bin_instances['memory_used'] >= task['memory_request'])
#             ]
#             if not eligible.empty:
#                 candidate = eligible.sort_values('price').iloc[0]
#                 candidate_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == candidate['instance_ID']][0]
#                 self.instance_bins.at[candidate_idx, 'bin_index'] = task_bin
#                 self._assign_task_to_instance(task, candidate)
#                 continue
#             # If no instance is found, mark as unscheduled
#             unscheduled_tasks.append(task)
#         return unscheduled_tasks

#     # -------------------------
#     # Scaling: Acquire New Instances for Unscheduled Tasks
#     # -------------------------
#     def scaler(self, unscheduled_tasks: List[pd.Series]):
#         tasks_by_bin = {}
#         for task in unscheduled_tasks:
#             bin_idx = self._calculate_bin_index(task['runtime'])
#             tasks_by_bin.setdefault(bin_idx, []).append(task)
#         for bin_idx in sorted(tasks_by_bin.keys(), reverse=True):
#             group = tasks_by_bin[bin_idx]
#             group.sort(key=lambda t: max(t['CPU_request'], t['memory_request']), reverse=True)
#             while group:
#                 best_score = -1
#                 best_instance_type = None
#                 best_group_size = 0
#                 for i in range(1, len(group) + 1):
#                     candidate_group = group[:i]
#                     for _, instance_type in self.available_instance_types.iterrows():
#                         if self._can_fit_group(candidate_group, instance_type):
#                             score = self._calculate_score(candidate_group, instance_type)
#                             if score > best_score:
#                                 best_score = score
#                                 best_instance_type = instance_type
#                                 best_group_size = i
#                 if best_instance_type is not None:
#                     new_instance = self._acquire_new_instance(bin_idx)
#                     for task in group[:best_group_size]:
#                         self._assign_task_to_instance(task, new_instance)
#                     group = group[best_group_size:]
#                 else:
#                     break

#     def _can_fit_group(self, task_group: List[pd.Series], instance_type: pd.Series) -> bool:
#         total_cpu = sum(task['CPU_request'] for task in task_group)
#         total_mem = sum(task['memory_request'] for task in task_group)
#         return (total_cpu <= instance_type['capacity_CPU'] * self.overbooking_factor and
#                 total_mem <= instance_type['capacity_memory'] * self.overbooking_factor)

#     def _calculate_score(self, task_group: List[pd.Series], instance_type: pd.Series) -> float:
#         total_cpu = sum(task['CPU_request'] for task in task_group)
#         total_mem = sum(task['memory_request'] for task in task_group)
#         utilization = (total_cpu + total_mem) / (instance_type['capacity_CPU'] + instance_type['capacity_memory'])
#         return utilization / instance_type['normalized_price']

#     # -------------------------
#     # Main Scheduling Method
#     # -------------------------
#     def schedule(self, new_tasks: pd.DataFrame) -> List[pd.Series]:
#         # 1. Try to pack tasks onto existing instances
#         unscheduled = self.packer(new_tasks)
#         # 2. Scale out for unscheduled tasks by grouping and acquiring new instances
#         self.scaler(unscheduled)
#         # 3. Migrate tasks from expensive instances to cheaper ones if possible
#         self._migrate_tasks()
#         # 4. Right-size idle instances to terminate underutilized resources
#         if not new_tasks.empty:
#             current_timestamp = new_tasks.iloc[0]['timestamp']
#             self._rightsize_instances(current_timestamp)
#         return []

#     # -------------------------
#     # Migration: Reassign Tasks from Expensive Instances
#     # -------------------------
#     def _migrate_tasks(self):
#         if self.instance_bins.empty:
#             return
#         min_price = self.instance_bins['price'].min()
#         expensive_instances = self.instance_bins[self.instance_bins['price'] > self.migration_threshold * min_price]
#         for idx, expensive in expensive_instances.iterrows():
#             tasks_on_instance = self.task_bins[self.task_bins['instance_ID'] == expensive['instance_ID']]
#             for _, task in tasks_on_instance.iterrows():
#                 bin_idx = task['bin_index']
#                 candidates = self.instance_bins[(self.instance_bins['bin_index'] == bin_idx) &
#                                                 (self.instance_bins['price'] < expensive['price'])]
#                 candidates = candidates.sort_values('price')
#                 for _, candidate in candidates.iterrows():
#                     if ((candidate['CPU_capacity'] - candidate['CPU_used'] >= task['CPU_request']) and
#                         (candidate['memory_capacity'] - candidate['memory_used'] >= task['memory_request'])):
#                         self._assign_task_to_instance(task, candidate)
#                         self.task_bins = self.task_bins[self.task_bins['instance_ID'] != expensive['instance_ID']]
#                         break

#     # -------------------------
#     # Neural Network Training
#     # -------------------------
#     def _train_multi_objective_network(self):
#         if len(self.replay_buffer.buffer) < self.batch_size:
#             return
#         experiences, is_weights = self.replay_buffer.sample(self.batch_size)
#         task_feats = torch.tensor(np.array([self._extract_task_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
#         inst_feats = torch.tensor(np.array([self._extract_instance_features(exp.instance_features) for exp in experiences]), dtype=torch.float32)
#         place_feats = torch.tensor(np.array([self._extract_placement_features() for _ in experiences]), dtype=torch.float32)
#         bin_feats = torch.tensor(np.array([self._extract_bin_features(exp.task_features.get('bin_index', 0)) for exp in experiences]), dtype=torch.float32)
#         temp_feats = torch.tensor(np.array([self._extract_temporal_features(exp.task_features) for exp in experiences]), dtype=torch.float32)
        
#         q_values, _, _, _ = self.policy_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
#         with torch.no_grad():
#             target_q_values, _, _, _ = self.target_network(task_feats, inst_feats, place_feats, bin_feats, temp_feats)
        
#         loss = ((q_values - target_q_values) ** 2 * torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1)).mean()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
#         self.training_steps += 1

#     # -------------------------
#     # Right-Sizing: Terminate Idle Instances
#     # -------------------------
#     def _rightsize_instances(self, current_timestamp):
#         idle_threshold = 0.05  # 5% utilization threshold
#         removable = []
#         for idx, instance in self.instance_bins.iterrows():
#             if instance['CPU_capacity'] > 0:
#                 utilization = instance['CPU_used'] / instance['CPU_capacity']
#             else:
#                 utilization = 0
#             if utilization < idle_threshold and (current_timestamp - instance['timestamp'] > instance['runtime']):
#                 removable.append(idx)
#         if removable:
#             self.instance_bins = self.instance_bins.drop(removable)
#             self.instance_counter -= len(removable)

#     # -------------------------
#     # Free Expired Tasks and Instances
#     # -------------------------
#     def free_tasks_and_instances(self, current_timestamp):
#         expired_tasks = self.task_bins[self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
#         expired_instances = self.instance_bins[self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum()
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum()
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,    # Learning rate
#         gamma: float = 0.9,    # Discount factor
#         epsilon: float = 0.1,  # Epsilon for epsilon-greedy
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that learns a policy to minimize cost and maximize
#         resource utilization over time.

#         Args:
#             available_instance_types: DataFrame with columns ['capacity_CPU','capacity_memory','normalized_price']
#             alpha: Q-learning learning rate
#             gamma: Discount factor for future rewards
#             epsilon: Probability of choosing a random action (exploration)
#             reward_scale: Scales the raw reward to control its magnitude
#         """

#         self.available_instance_types = available_instance_types

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # Track tasks and instances (similar to other schedulers)
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.reward_scale = reward_scale

#         # Q-table: Maps (state, action) -> Q-value
#         # In practice, you may need a more advanced structure (e.g., dict-of-dict) or function approximator
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp,
#         similar to your other schedulers.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main entry point for scheduling. For each task:
#         1. Construct a state representation.
#         2. Choose an action (existing instance or new instance type).
#         3. Execute that action (assign the task).
#         4. Calculate reward and update Q-table.
#         """
#         # Convert the DataFrame to a list of tasks for convenience
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             # 1. Observe current state
#             state = self._get_state_representation(task)

#             # 2. Get possible actions for this task
#             possible_actions = self._get_possible_actions(task)

#             # 3. Choose action via epsilon-greedy
#             action = self._choose_action(state, possible_actions)

#             # 4. Execute the chosen action, get immediate reward and next_state
#             next_state, reward = self._execute_action(task, action)

#             # 5. Update Q-table
#             self._update_q_table(state, action, reward, next_state)

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Construct a simplified state representation.

#         Example:
#           state = (
#             total_active_instances (discretized),
#             total_cpu_used (discretized),
#             total_memory_used (discretized),
#             task_cpu_request_bin,
#             task_memory_request_bin
#           )

#         You can customize/discretize as needed.
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         # Simple binning for CPU/memory usage
#         cpu_bin = int(total_cpu_used // 50)  # example bin size
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         # Bin the task's CPU/memory request
#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         actions = []

#         # 1) Check existing instances that can fit the task
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) Check all instance types for a new instance using positional indices
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # Fallback action if no actions are possible
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

    
#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         # Random exploration: use built-in random.choice instead of np.random.choice
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploitation: pick best action from Q-table
#         q_values = []
#         for action in possible_actions:
#             q_values.append(self.q_table.get((state, action), 0.0))

#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Perform the chosen action and return (next_state, reward).

#         action is either:
#           ("use_existing", instance_id)  or
#           ("new_instance", instance_type_idx)

#         Reward is computed based on changes in cost/utilization, etc.
#         """
#         action_type, action_value = action

#         # Current cost, utilization before scheduling
#         old_price = self.price_counter
#         old_cpu_util = self.cpu_utilization
#         old_mem_util = self.memory_utilization

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()

#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # If action is "unscheduled" or something else
#             # We won't schedule the task (penalize reward).
#             pass

#         # Next state
#         next_state = self._get_state_representation(task)

#         # Compute immediate reward
#         # Example: negative cost + positive utilization gain
#         new_price = self.price_counter
#         new_cpu_util = self.cpu_utilization
#         new_mem_util = self.memory_utilization

#         cost_delta = new_price - old_price
#         cpu_util_delta = new_cpu_util - old_cpu_util
#         mem_util_delta = new_mem_util - old_mem_util

#         # A simple example reward:
#         #   reward = -cost_delta + (cpu_util_delta + mem_util_delta)/2
#         # Scale or adjust as needed
#         reward = (-cost_delta) + 0.1 * (cpu_util_delta + mem_util_delta)
#         reward *= self.reward_scale

#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Standard Q-learning update:
#           Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         # Next-state max Q
#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         if next_q_values:
#             max_next_q = max(next_q_values)
#         else:
#             max_next_q = 0.0

#         # Update
#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         Optional: generate all possible actions for the given state if needed.
#         For simplicity, we can return an empty list here, or replicate logic from `_get_possible_actions`.
#         In a full multi-step approach, you'd consider all possible tasks that could arrive.
#         """
#         # Typically, you'd want to do something more robust if you do multi-step lookahead.
#         return []

#     # -------------------------
#     # HELPER METHODS (similar to other schedulers)
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Calculate bin index based on runtime (same approach as your other schedulers)."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Similar logic to your other schedulers: updates instance resources,
#         updates cost, etc.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add to task_bins
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
#         """
#         Similar to your other schedulers. Acquire a new instance of the given type,
#         add to instance_bins, update utilization, etc.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update self.cpu_utilization and self.memory_utilization based on
#         total capacity vs. total used across all instances.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,     # Learning rate
#         gamma: float = 0.9,     # Discount factor
#         epsilon: float = 0.3,   # Starting epsilon for exploration
#         epsilon_min: float = 0.01,  # Minimum epsilon after decay
#         epsilon_decay: float = 0.99, # Epsilon decay factor
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that learns a policy to minimize cost
#         and maximize resource utilization over time.

#         Changes from the basic version:
#           - Epsilon decay: the agent explores less over time.
#           - Modified reward function: places more emphasis on CPU/mem utilization.

#         Args:
#             available_instance_types: DataFrame with columns:
#                 ['capacity_CPU','capacity_memory','normalized_price']
#             alpha: Q-learning learning rate
#             gamma: Discount factor for future rewards
#             epsilon: Initial probability of random exploration
#             epsilon_min: Minimum value for epsilon
#             epsilon_decay: Factor by which epsilon is multiplied each episode/iteration
#             reward_scale: Scales the raw reward to control its magnitude
#         """

#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # Track tasks and instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Q-table: (state, action) -> Q-value
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main entry point for scheduling. For each task:
#         1. Construct a state representation.
#         2. Choose an action (existing instance or new instance type).
#         3. Execute that action (assign the task).
#         4. Calculate reward and update Q-table.
#         5. Decay epsilon (optional per-task or per-timestamp).
#         """
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             # 1. Observe current state
#             state = self._get_state_representation(task)

#             # 2. Get possible actions for this task
#             possible_actions = self._get_possible_actions(task)

#             # 3. Choose action via epsilon-greedy
#             action = self._choose_action(state, possible_actions)

#             # 4. Execute action, get next_state + reward
#             next_state, reward = self._execute_action(task, action)

#             # 5. Update Q-table
#             self._update_q_table(state, action, reward, next_state)

#             # Optionally decay epsilon after each task
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Example state:
#           (
#             num_active_instances,
#             total_CPU_used // 50,
#             total_MEM_used // 50,
#             task_CPU_req // 10,
#             task_MEM_req // 10
#           )
#         Customize/discretize as needed.
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions:
#           ("use_existing", instance_id) if an instance can fit the task
#           ("new_instance", i) for each instance type that can fit the task
#           ("unscheduled", None) if no feasible action found
#         """
#         actions = []

#         # 1) Existing instances
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) New instance (positional index)
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # 3) Fallback if no action possible
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy. Use random.choice for exploration, or pick the best Q-value.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploitation
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the chosen action. Return (next_state, reward).
#         """
#         action_type, action_value = action

#         # Record old cost/utilization
#         old_price = self.price_counter
#         old_cpu_util = self.cpu_utilization
#         old_mem_util = self.memory_utilization

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()

#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # "unscheduled" or any fallback
#             pass

#         # Compute next state
#         next_state = self._get_state_representation(task)

#         # Compute immediate reward
#         new_price = self.price_counter
#         new_cpu_util = self.cpu_utilization
#         new_mem_util = self.memory_utilization

#         cost_delta = new_price - old_price
#         cpu_util_delta = new_cpu_util - old_cpu_util
#         mem_util_delta = new_mem_util - old_mem_util

#         # Emphasize cost reduction & resource utilization
#         # You can adjust these coefficients to see different behaviors
#         reward = -cost_delta + 0.5 * (cpu_util_delta + mem_util_delta)
#         reward *= self.reward_scale

#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Q-learning update:
#           Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         # Next-state max Q
#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         max_next_q = max(next_q_values) if next_q_values else 0.0

#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         If you want multi-step lookahead, you might replicate _get_possible_actions
#         for 'next_state' tasks. For now, we return an empty list or minimal approach.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Same approach as your other schedulers for binning runtime."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Similar logic to your other schedulers: updates instance resources, cost, etc.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add to task_bins
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
#         """
#         Acquire a new instance of the given type, add to instance_bins.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update cpu_utilization and memory_utilization.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to shift from exploration to exploitation.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,     # Learning rate
#         gamma: float = 0.9,     # Discount factor
#         epsilon: float = 0.3,   # Starting epsilon for exploration
#         epsilon_min: float = 0.01,  # Minimum epsilon after decay
#         epsilon_decay: float = 0.99, # Epsilon decay factor
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that focuses on reducing total cost as much as possible,
#         with CPU and memory utilization as a secondary concern.

#         Key changes from previous version:
#           - Heavier penalty for cost increases in the reward function.
#           - Utilization has a much smaller weight, making cost the primary optimization target.
#         """

#         # Reset index so enumerating instance types matches .iloc indexing
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # Track tasks and instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Q-table: (state, action) -> Q-value
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main entry point for scheduling. For each task:
#         1. Construct a state representation.
#         2. Choose an action (existing instance or new instance type).
#         3. Execute that action (assign the task).
#         4. Calculate reward and update Q-table.
#         5. Decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             # 1. Observe current state
#             state = self._get_state_representation(task)

#             # 2. Get possible actions for this task
#             possible_actions = self._get_possible_actions(task)

#             # 3. Choose action via epsilon-greedy
#             action = self._choose_action(state, possible_actions)

#             # 4. Execute action, get next_state + reward
#             next_state, reward = self._execute_action(task, action)

#             # 5. Update Q-table
#             self._update_q_table(state, action, reward, next_state)

#             # Decay epsilon after each task
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Example state:
#           (
#             num_active_instances,
#             total_CPU_used // 50,
#             total_MEM_used // 50,
#             task_CPU_req // 10,
#             task_MEM_req // 10
#           )
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions:
#           ("use_existing", instance_id) if an instance can fit the task
#           ("new_instance", i) for each instance type that can fit the task
#           ("unscheduled", None) if no feasible action found
#         """
#         actions = []

#         # 1) Existing instances
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) New instance (positional index)
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # 3) Fallback if no action possible
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy. Use random.choice for exploration, or pick the best Q-value.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploitation
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the chosen action. Return (next_state, reward).
#         """
#         action_type, action_value = action

#         # Record old cost/utilization
#         old_price = self.price_counter
#         old_cpu_util = self.cpu_utilization
#         old_mem_util = self.memory_utilization

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()

#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # "unscheduled" or any fallback
#             pass

#         # Compute next state
#         next_state = self._get_state_representation(task)

#         # Compute immediate reward
#         new_price = self.price_counter
#         new_cpu_util = self.cpu_utilization
#         new_mem_util = self.memory_utilization

#         cost_delta = new_price - old_price
#         cpu_util_delta = new_cpu_util - old_cpu_util
#         mem_util_delta = new_mem_util - old_mem_util

#         # Heavily penalize cost, small positive weight for utilization
#         # If cost_delta is large, it will strongly reduce the reward.
#         # If cost doesn't change but utilization goes up, we get a small reward.
#         reward = (-cost_delta) + 0.05 * (cpu_util_delta + mem_util_delta)
#         reward *= self.reward_scale

#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Q-learning update:
#           Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         # Next-state max Q
#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         max_next_q = max(next_q_values) if next_q_values else 0.0

#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         If you want multi-step lookahead, you might replicate _get_possible_actions
#         for 'next_state' tasks. For now, we return an empty list or minimal approach.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Same approach as your other schedulers for binning runtime."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Updates instance resources and cost, similarly to your other schedulers.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add to task_bins
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
#         """
#         Acquire a new instance of the given type, add to instance_bins.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update cpu_utilization and memory_utilization.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to shift from exploration to exploitation.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,       # Learning rate
#         gamma: float = 0.9,       # Discount factor
#         epsilon: float = 0.3,     # Starting epsilon for exploration
#         epsilon_min: float = 0.01,# Minimum epsilon after decay
#         epsilon_decay: float = 0.99, # Epsilon decay factor
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that *strictly* prioritizes cost reduction.
#         CPU/memory utilization is effectively a low-priority concern.

#         Key differences from previous versions:
#           - Reward function is dominated by cost penalty.
#           - If cost goes up, the reward is strongly negative.
#           - CPU/memory utilization only has a tiny positive effect (or none at all).
#         """

#         # Reset index to ensure .iloc references match enumerated indices
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks and instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Q-table: (state, action) -> Q-value
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main entry point for scheduling. For each task:
#           1. Construct a state representation.
#           2. Get possible actions.
#           3. Choose action (epsilon-greedy).
#           4. Execute action, compute reward.
#           5. Update Q-table.
#           6. Decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             state = self._get_state_representation(task)
#             possible_actions = self._get_possible_actions(task)
#             action = self._choose_action(state, possible_actions)
#             next_state, reward = self._execute_action(task, action)
#             self._update_q_table(state, action, reward, next_state)
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Example coarse state:
#           (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin)
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions:
#           ("use_existing", instance_id)
#           ("new_instance", i)
#           ("unscheduled", None)
#         """
#         actions = []

#         # 1) Existing instances that can fit the task
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) Acquire new instance if it fits
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploit
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the chosen action. Return (next_state, reward).
#         """
#         action_type, action_value = action

#         # Track old cost
#         old_price = self.price_counter

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()
#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # "unscheduled" -> no scheduling, no cost change
#             pass

#         # Next state
#         next_state = self._get_state_representation(task)

#         # Updated cost
#         new_price = self.price_counter
#         cost_delta = new_price - old_price

#         # For strict cost optimization: strongly penalize cost changes.
#         # We'll ignore CPU/memory utilization entirely, or you can keep a tiny factor if you want.
#         # e.g. reward = -10 * cost_delta
#         # If cost_delta is big, reward is very negative.
#         reward = -10.0 * cost_delta
#         reward *= self.reward_scale

#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Standard Q-learning update.
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         max_next_q = max(next_q_values) if next_q_values else 0.0

#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         If you want multi-step lookahead, replicate logic from _get_possible_actions
#         for the next state's tasks. For now, it's empty or minimal.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Same approach as your other schedulers for binning runtime."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Similar logic to your other schedulers: updates instance resources, cost, etc.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add to task_bins
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
#         """
#         Acquire a new instance of the given type, add to instance_bins.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update cpu_utilization and memory_utilization, though we don't use it in the reward.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to shift from exploration to exploitation.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,         # Learning rate
#         gamma: float = 0.9,         # Discount factor
#         epsilon: float = 0.3,       # Starting epsilon for exploration
#         epsilon_min: float = 0.01,  # Minimum epsilon after decay
#         epsilon_decay: float = 0.99,# Epsilon decay factor
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that *strictly* prioritizes cost reduction.
#         Now with additional penalties for acquiring new instances to ensure minimal cost.
#         """

#         # Reset index to ensure .iloc references match enumerated indices
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks and instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Q-table: (state, action) -> Q-value
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main entry point for scheduling. For each task:
#           1. Construct a state representation.
#           2. Get possible actions.
#           3. Choose action (epsilon-greedy).
#           4. Execute action, compute reward.
#           5. Update Q-table.
#           6. Decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             state = self._get_state_representation(task)
#             possible_actions = self._get_possible_actions(task)
#             action = self._choose_action(state, possible_actions)
#             next_state, reward = self._execute_action(task, action)
#             self._update_q_table(state, action, reward, next_state)
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Example coarse state:
#           (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin)
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions:
#           ("use_existing", instance_id)
#           ("new_instance", i)
#           ("unscheduled", None)
#         """
#         actions = []

#         # 1) Existing instances that can fit the task
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) Acquire new instance if it fits
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # 3) Optionally, you can keep an "unscheduled" action (though typically we want to schedule).
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploit
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the chosen action. Return (next_state, reward).
#         """
#         action_type, action_value = action

#         # Track old cost
#         old_price = self.price_counter

#         # Flag for new instance penalty
#         new_instance_penalty = 0.0

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()
#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]

#             # Penalize acquiring a new instance (especially if it's expensive).
#             # This helps discourage unnecessary instance creation.
#             new_instance_penalty = inst_type['normalized_price'] * 10.0

#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # "unscheduled" -> no scheduling, no cost change
#             pass

#         # Next state
#         next_state = self._get_state_representation(task)

#         # Updated cost
#         new_price = self.price_counter
#         cost_delta = new_price - old_price

#         # Strongly penalize increases in cost
#         # You can tune the factor (e.g., 50, 100) depending on how strongly you want to penalize cost
#         reward = -100.0 * cost_delta

#         # Add penalty for acquiring a new instance
#         reward -= new_instance_penalty

#         # Optionally scale the reward further
#         reward *= self.reward_scale

#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Standard Q-learning update.
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         max_next_q = max(next_q_values) if next_q_values else 0.0

#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         If you want multi-step lookahead, replicate logic from _get_possible_actions
#         for the next state's tasks. For now, it's minimal.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Same approach as your other schedulers for binning runtime."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Updates instance resources, cost, etc.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add to task_bins
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
#         """
#         Acquire a new instance of the given type, add to instance_bins.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update cpu_utilization and memory_utilization (not used in reward but tracked).
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to shift from exploration to exploitation.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,          # Learning rate
#         gamma: float = 0.9,          # Discount factor
#         epsilon: float = 0.3,        # Starting epsilon for exploration
#         epsilon_min: float = 0.01,   # Minimum epsilon after decay
#         epsilon_decay: float = 0.95, # Faster epsilon decay for quicker convergence
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning–based scheduler that prioritizes cost reduction.
#         Further modifications:
#           - Cost increases are now heavily penalized.
#           - Creating new instances is punished more severely.
#           - Reusing an instance provides a small bonus.
#         """
#         # Reset index to ensure .iloc references match enumerated indices
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks and instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])

#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Q-table: (state, action) -> Q-value
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop. For each task:
#           1. Build the state.
#           2. Determine possible actions.
#           3. Choose an action via epsilon-greedy.
#           4. Execute the action and compute the reward.
#           5. Update the Q-table.
#           6. Decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')

#         for task in task_list:
#             state = self._get_state_representation(task)
#             possible_actions = self._get_possible_actions(task)
#             action = self._choose_action(state, possible_actions)
#             next_state, reward = self._execute_action(task, action)
#             self._update_q_table(state, action, reward, next_state)
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Coarse state representation:
#           (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin)
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)

#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         state = (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin
#         )
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Returns possible actions:
#           ("use_existing", instance_id)
#           ("new_instance", i)
#           ("unscheduled", None)
#         """
#         actions = []

#         # 1) Use existing instances if they have sufficient capacity
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) Option to acquire a new instance
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # 3) If no action is available, mark as unscheduled.
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy action selection.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)

#         # Exploitation: choose action with highest Q-value.
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the action and compute the reward.
#         """
#         action_type, action_value = action

#         # Record the current cost before executing the action.
#         old_price = self.price_counter
#         new_instance_penalty = 0.0
#         reuse_bonus = 0.0

#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == instance_id
#             ].squeeze()
#             # Bonus for using an existing instance.
#             reuse_bonus = 50.0
#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             # Apply a heavy penalty for acquiring a new instance.
#             new_instance_penalty = inst_type['normalized_price'] * 100.0
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # "unscheduled" case
#             pass

#         next_state = self._get_state_representation(task)
#         new_price = self.price_counter
#         cost_delta = new_price - old_price

#         # Strong cost penalty plus bonus/penalties based on action type.
#         reward = -1000.0 * cost_delta
#         reward -= new_instance_penalty
#         reward += reuse_bonus

#         reward *= self.reward_scale
#         return next_state, reward

#     def _update_q_table(
#         self,
#         state: Tuple,
#         action: Tuple,
#         reward: float,
#         next_state: Tuple
#     ):
#         """
#         Standard Q-learning update.
#         """
#         old_q = self.q_table.get((state, action), 0.0)

#         next_q_values = [
#             self.q_table.get((next_state, a), 0.0)
#             for a in self._all_possible_actions(next_state)
#         ]
#         max_next_q = max(next_q_values) if next_q_values else 0.0

#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         Minimal lookahead: no simulated actions for the next state.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Binning of runtime as used in the other schedulers."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Update instance resources and cost when assigning a task.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

#         # Update utilization metrics
#         self._update_utilization()

#         # Price calculation logic (using runtime and price)
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

#         # Add the task to the task_bins log
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
#         """
#         Acquire and register a new instance.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': instance_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': instance_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': instance_type['normalized_price']
#         })

#         self.instance_bins = pd.concat(
#             [self.instance_bins, pd.DataFrame([new_instance])],
#             ignore_index=True
#         )

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update CPU and memory utilization metrics.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Reduce epsilon gradually to favor exploitation over exploration.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,          # Learning rate
#         gamma: float = 0.9,          # Discount factor
#         epsilon: float = 0.3,        # Starting epsilon for exploration
#         epsilon_min: float = 0.01,   # Minimum epsilon after decay
#         epsilon_decay: float = 0.95, # Faster decay for quicker convergence
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning scheduler that aggressively optimizes for cost reduction.
#         Modifications include:
#          - An enriched state representation (including discretized current cost)
#          - A refined reward function with quadratic cost penalties for increases,
#            dynamic new-instance penalties, and bonuses for reusing instances.
#         """
#         # Reset index so that .iloc references match enumerated indices.
#         self.available_instance_types = available_instance_types.reset_index(drop=True)
#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()
        
#         # DataFrames for tasks and instances.
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price', 'instance_type'
#         ])
        
#         # Counters.
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0  # Total cost so far.

#         self.instance_counters = [0] * 10

        
#         # Utilization metrics.
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0
        
#         # Q-learning parameters.
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale
        
#         # Q-table: mapping (state, action) -> Q-value.
#         self.q_table: Dict[Tuple, float] = {}

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free tasks that have expired.
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
        
#         # Free expired instances.
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         for _, instance in expired_instances.iterrows():
#             instance_type_index = int(instance['instance_type'])
#             self.instance_counters[instance_type_index] -= 1
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        
#         # Update utilization metrics.
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop: for each task, build the state, get actions, choose an action,
#         execute it, update the Q-table, and decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')
#         for task in task_list:
#             state = self._get_state_representation(task)
#             possible_actions = self._get_possible_actions(task)
#             action = self._choose_action(state, possible_actions)
#             next_state, reward = self._execute_action(task, action)
#             self._update_q_table(state, action, reward, next_state)
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------
    
#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         State representation now includes:
#           (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
#         where cost_bin is a discretized form of the current total cost.
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()
        
#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)
        
#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)
        
#         # Discretize current total cost.
#         cost_bin = int(self.price_counter // 50)
        
#         state = (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions include:
#           ("use_existing", instance_id)
#           ("new_instance", i)
#           ("unscheduled", None)
#         """
#         actions = []
#         # 1) Actions to use an existing instance (if enough capacity).
#         for _, inst in self.instance_bins.iterrows():
#             if (inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']):
#                 actions.append(("use_existing", int(inst['instance_ID'])))
#         # 2) Actions to acquire a new instance.
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']):
#                 actions.append(("new_instance", i))
#         # 3) Fallback: if no valid action, mark as unscheduled.
#         if not actions:
#             actions.append(("unscheduled", None))
#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Choose an action using an epsilon-greedy policy.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)
#         # Exploitation: choose the action with highest Q-value.
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Execute the chosen action and compute the reward.
#         The reward function uses a quadratic penalty for cost increases,
#         dynamic penalties for new instance creation, and a bonus for reusing instances.
#         """
#         action_type, action_value = action
#         old_price = self.price_counter
#         new_instance_penalty = 0.0
#         reuse_bonus = 0.0
        
#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
#             reuse_bonus = 100.0  # Bonus for reusing an instance.
#             self._assign_task_to_instance(task, instance)
        
#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             # Heavier dynamic penalty for acquiring a new instance.
#             new_instance_penalty = inst_type['normalized_price'] * 200.0
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)
        
#         else:  # "unscheduled" action.
#             new_instance_penalty = 50.0  # Small penalty for leaving task unscheduled.
        
#         next_state = self._get_state_representation(task)
#         new_price = self.price_counter
#         cost_delta = new_price - old_price
        
#         # Apply a quadratic penalty when cost increases to heavily discourage any increase.
#         if cost_delta > 0:
#             cost_penalty = 1000.0 * (cost_delta ** 2)
#         else:
#             cost_penalty = 1000.0 * cost_delta  # Linear reward for cost decreases.
        
#         reward = -cost_penalty - new_instance_penalty + reuse_bonus
#         reward *= self.reward_scale
#         return next_state, reward

#     def _update_q_table(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
#         """
#         Update the Q-table using the standard Q-learning update rule.
#         """
#         old_q = self.q_table.get((state, action), 0.0)
#         next_q_values = [self.q_table.get((next_state, a), 0.0)
#                          for a in self._all_possible_actions(next_state)]
#         max_next_q = max(next_q_values) if next_q_values else 0.0
#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         For multi-step lookahead. Here we return an empty list for simplicity.
#         """
#         return []

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------
    
#     def _calculate_bin_index(self, runtime: float) -> int:
#         """
#         Bin the runtime similarly to other schedulers.
#         """
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Update the instance's resource usage and cost when assigning a task.
#         """
#         instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']
        
#         # Update utilization metrics.
#         self._update_utilization()
        
#         # Price calculation: if this is the first task on the instance.
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime
        
#         # Log the task assignment.
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
#         """
#         Acquire a new instance of the given type and register it.
#         """
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
#         self.instance_counters[instance_type_num] += 1
#         self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update CPU and memory utilization metrics.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to favor exploitation over exploration.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class QLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,          # Learning rate
#         gamma: float = 0.9,          # Discount factor
#         epsilon: float = 0.3,        # Starting epsilon for exploration
#         epsilon_min: float = 0.01,   # Minimum epsilon after decay
#         epsilon_decay: float = 0.95, # Faster decay for quicker convergence
#         reward_scale: float = 1.0
#     ):
#         """
#         A Q-learning scheduler designed to aggressively reduce costs.
#         Modifications include:
#          - Enhanced state representation: includes current cost and utilization.
#          - A refined reward function that:
#               * Applies an exponential penalty for cost increases.
#               * Heavily penalizes new instance creation (multiplier 500× normalized_price).
#               * Gives a boosted bonus for reusing existing instances.
#               * Adds an extra bonus when overall resource utilization is very high.
#          - A non-empty lookahead to guide Q-value updates.
#         """
#         self.available_instance_types = available_instance_types.reset_index(drop=True)
#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()
        
#         # DataFrames for tasks and instances.
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price', 'instance_type'
#         ])
        
#         # Counters.
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0  # Total cost so far.
#         self.instance_counters = [0] * 10
        
#         # Utilization metrics.
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0
        
#         # Q-learning parameters.
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale
        
#         # Q-table: mapping (state, action) -> Q-value.
#         self.q_table: Dict[Tuple, float] = {}

#     def reset(self):
#         """
#         Resets the scheduler's environment state for a new episode while preserving the Q-table.
#         """
#         self.task_bins = self.task_bins.iloc[0:0]
#         self.instance_bins = self.instance_bins.iloc[0:0]
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0
#         self.instance_counters = [0] * 10
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         """
#         # Free tasks that have expired.
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             instance_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
        
#         # Free expired instances.
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         for _, instance in expired_instances.iterrows():
#             instance_type_index = int(instance['instance_type'])
#             self.instance_counters[instance_type_index] -= 1
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        
#         # Update utilization metrics.
#         self._update_utilization()

#     def q_learning_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop: For each task, construct the state, select and execute an action,
#         update the Q-table, and decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')
#         for task in task_list:
#             state = self._get_state_representation(task)
#             possible_actions = self._get_possible_actions(task)
#             action = self._choose_action(state, possible_actions)
#             next_state, reward = self._execute_action(task, action)
#             self._update_q_table(state, action, reward, next_state)
#             self._decay_epsilon()

#     # -------------------------
#     # INTERNAL Q-LEARNING LOGIC
#     # -------------------------
    
#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Enhanced state representation includes:
#          (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin, cpu_util, mem_util)
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()
        
#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)
        
#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)
        
#         cost_bin = int(self.price_counter // 50)
        
#         state = (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin,
#                  round(self.cpu_utilization, 1), round(self.memory_utilization, 1))
#         return state

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Returns possible actions:
#          - ("use_existing", instance_id)
#          - ("new_instance", i)
#          - ("unscheduled", None)
#         """
#         actions = []
#         # Option 1: Use existing instance if capacity allows.
#         for _, inst in self.instance_bins.iterrows():
#             if (inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']):
#                 actions.append(("use_existing", int(inst['instance_ID'])))
#         # Option 2: Acquire a new instance.
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']):
#                 actions.append(("new_instance", i))
#         # Option 3: Fallback unscheduled action.
#         if not actions:
#             actions.append(("unscheduled", None))
#         return actions

#     def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy: explore or choose the best known action.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(possible_actions)
#         q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
#         best_action_idx = int(np.argmax(q_values))
#         return possible_actions[best_action_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Executes the chosen action and computes the reward.
#         Uses an exponential penalty for cost increases and further incentivizes instance reuse.
#         """
#         action_type, action_value = action
#         old_price = self.price_counter
#         new_instance_penalty = 0.0
#         reuse_bonus = 0.0
        
#         if action_type == "use_existing":
#             instance_id = action_value
#             instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
#             reuse_bonus = 200.0  # Increased bonus for reusing an instance.
#             self._assign_task_to_instance(task, instance)
        
#         elif action_type == "new_instance":
#             instance_type_idx = action_value
#             inst_type = self.available_instance_types.iloc[instance_type_idx]
#             new_instance_penalty = inst_type['normalized_price'] * 500.0  # Heavier penalty for new instance.
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)
        
#         else:  # "unscheduled" action.
#             new_instance_penalty = 100.0  # Higher penalty for not scheduling.
        
#         next_state = self._get_state_representation(task)
#         new_price = self.price_counter
#         cost_delta = new_price - old_price
        
#         # Exponential penalty for cost increases; linear reward for cost decreases.
#         if cost_delta > 0:
#             cost_penalty = 1000.0 * (np.exp(cost_delta) - 1)
#         else:
#             cost_penalty = 1000.0 * cost_delta
        
#         # Utilization bonus: if average utilization exceeds thresholds, reward more.
#         avg_util = (self.cpu_utilization + self.memory_utilization) / 2.0
#         if avg_util > 85:
#             utilization_bonus = 100.0
#         elif avg_util > 75:
#             utilization_bonus = 50.0
#         else:
#             utilization_bonus = 0.0
        
#         reward = -cost_penalty - new_instance_penalty + reuse_bonus + utilization_bonus
#         reward *= self.reward_scale
#         return next_state, reward

#     def _update_q_table(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
#         """
#         Standard Q-learning update rule.
#         """
#         old_q = self.q_table.get((state, action), 0.0)
#         next_possible_actions = self._all_possible_actions(next_state)
#         next_q_values = [self.q_table.get((next_state, a), 0.0) for a in next_possible_actions]
#         max_next_q = max(next_q_values) if next_q_values else 0.0
#         new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
#         self.q_table[(state, action)] = new_q

#     def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
#         """
#         Lookahead: generate a dummy task from the state to determine possible actions.
#         """
#         dummy_task = {
#             'CPU_request': state[3] * 10,
#             'memory_request': state[4] * 10,
#             'runtime': 1,
#             'timestamp': 0,
#             'job_ID': None,
#             'task_index': None
#         }
#         return self._get_possible_actions(dummy_task)

#     # -------------------------
#     # HELPER METHODS
#     # -------------------------
    
#     def _calculate_bin_index(self, runtime: float) -> int:
#         """
#         Bin the runtime similarly to other schedulers.
#         """
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Update instance resource usage and cost when assigning a task.
#         """
#         instance_idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]
#         self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']
        
#         # Update utilization metrics.
#         self._update_utilization()
        
#         # Price calculation: if first task on instance.
#         if self.instance_bins.at[instance_idx, 'runtime'] == 0:
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
#         else:
#             max_timestamp = max(
#                 task['runtime'] + task['timestamp'],
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             additional_runtime = (task['runtime'] + task['timestamp']) - (
#                 self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
#             )
#             self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']
#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime
        
#         # Log task assignment.
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
#         """
#         Acquire and register a new instance.
#         """
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
#         self.instance_counters[instance_type_num] += 1
#         self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Updates CPU and memory utilization metrics.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduce epsilon to favor exploitation over exploration.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import random

class QLearningScheduler:
    def __init__(
        self,
        available_instance_types: pd.DataFrame,
        alpha: float = 0.1,          # Learning rate
        gamma: float = 0.9,          # Discount factor
        epsilon: float = 0.3,        # Starting epsilon for exploration
        epsilon_min: float = 0.01,   # Minimum epsilon after decay
        epsilon_decay: float = 0.95, # Faster decay for quicker convergence
        reward_scale: float = 1.0
    ):
        """
        A Q-learning scheduler that aggressively optimizes for cost reduction.
        Every task is scheduled; there is no unscheduled option.
        Modifications include:
         - An enriched state representation (including discretized current cost).
         - A refined reward function with quadratic cost penalties for increases,
           dynamic new-instance penalties, and bonuses for reusing instances.
         - The unscheduled option has been removed so that every task is assigned.
        """
        # Reset index so that .iloc references match enumerated indices.
        self.available_instance_types = available_instance_types.reset_index(drop=True)
        self.max_cpu = self.available_instance_types['capacity_CPU'].max()
        self.max_memory = self.available_instance_types['capacity_memory'].max()
        
        # DataFrames for tasks and instances.
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 'task_index', 'bin_index', 'instance_ID',
            'CPU_request', 'memory_request', 'timestamp', 'runtime'
        ])
        self.instance_bins = pd.DataFrame(columns=[
            'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
            'memory_capacity', 'memory_used', 'timestamp', 'runtime',
            'price', 'instance_type'
        ])
        
        # Counters.
        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.price_counter = 0.0  # Total cost so far.
        self.instance_counters = [0] * 10
        
        # Utilization metrics.
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        
        # Q-learning parameters.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale
        
        # Q-table: mapping (state, action) -> Q-value.
        self.q_table: Dict[Tuple, float] = {}

    def reset(self):
        """
        Resets the scheduler's environment state for a new episode while preserving the Q-table.
        """
        self.task_bins = self.task_bins.iloc[0:0]
        self.instance_bins = self.instance_bins.iloc[0:0]
        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.price_counter = 0.0
        self.instance_counters = [0] * 10
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0

    def free_tasks_and_instances(self, current_timestamp: float):
        """
        Frees expired tasks and instances at the given timestamp.
        """
        # Free tasks that have expired.
        expired_tasks = self.task_bins[
            self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
        ]
        for _, task in expired_tasks.iterrows():
            instance_id = task['instance_ID']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
        
        # Free expired instances.
        expired_instances = self.instance_bins[
            self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
        ]
        for _, instance in expired_instances.iterrows():
            instance_type_index = int(instance['instance_type'])
            self.instance_counters[instance_type_index] -= 1
        self.instance_counter -= len(expired_instances)
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        
        # Update utilization metrics.
        self._update_utilization()

    def q_learning_scheduler(self, new_tasks: pd.DataFrame):
        """
        Main scheduling loop: for each task, build the state, get actions, choose an action,
        execute it, update the Q-table, and decay epsilon.
        """
        task_list = new_tasks.to_dict('records')
        for task in task_list:
            state = self._get_state_representation(task)
            possible_actions = self._get_possible_actions(task)
            action = self._choose_action(state, possible_actions)
            next_state, reward = self._execute_action(task, action)
            self._update_q_table(state, action, reward, next_state)
            self._decay_epsilon()

    # -------------------------
    # INTERNAL Q-LEARNING LOGIC
    # -------------------------
    
    def _get_state_representation(self, task: Dict) -> Tuple:
        """
        State representation now includes:
          (num_active_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
        where cost_bin is a discretized form of the current total cost.
        """
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_mem_used = self.instance_bins['memory_used'].sum()
        
        cpu_bin = int(total_cpu_used // 50)
        mem_bin = int(total_mem_used // 50)
        num_instances = len(self.instance_bins)
        
        task_cpu_bin = int(task['CPU_request'] // 10)
        task_mem_bin = int(task['memory_request'] // 10)
        
        # Discretize current total cost.
        cost_bin = int(self.price_counter // 50)
        
        state = (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
        return state

    def _get_possible_actions(self, task: Dict) -> List[Tuple]:
        """
        Possible actions include:
          ("use_existing", instance_id)
          ("new_instance", i)
        There is no unscheduled option.
        """
        actions = []
        # 1) Actions to use an existing instance (if enough capacity).
        for _, inst in self.instance_bins.iterrows():
            if (inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
                inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']):
                actions.append(("use_existing", int(inst['instance_ID'])))
        # 2) Actions to acquire a new instance.
        for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
            if (inst_type['capacity_CPU'] >= task['CPU_request'] and
                inst_type['capacity_memory'] >= task['memory_request']):
                actions.append(("new_instance", i))
        # Force scheduling: if no action is available, force new instance creation using the first available type.
        if not actions:
            actions.append(("new_instance", 0))
        return actions

    def _choose_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
        """
        Choose an action using an epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return random.choice(possible_actions)
        # Exploitation: choose the action with highest Q-value.
        q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
        best_action_idx = int(np.argmax(q_values))
        return possible_actions[best_action_idx]

    def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
        """
        Execute the chosen action and compute the reward.
        The reward function uses a quadratic penalty for cost increases,
        dynamic penalties for new instance creation, and a bonus for reusing instances.
        """
        action_type, action_value = action
        old_price = self.price_counter
        new_instance_penalty = 0.0
        reuse_bonus = 0.0
        
        if action_type == "use_existing":
            instance_id = action_value
            instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
            reuse_bonus = 100.0  # Bonus for reusing an instance.
            self._assign_task_to_instance(task, instance)
        
        elif action_type == "new_instance":
            instance_type_idx = action_value
            inst_type = self.available_instance_types.iloc[instance_type_idx]
            # Heavier dynamic penalty for acquiring a new instance.
            new_instance_penalty = inst_type['normalized_price'] * 200.0
            new_inst = self._acquire_new_instance(
                inst_type,
                self._calculate_bin_index(task['runtime'])
            )
            self._assign_task_to_instance(task, new_inst)
        
        next_state = self._get_state_representation(task)
        new_price = self.price_counter
        cost_delta = new_price - old_price
        
        # Apply a quadratic penalty when cost increases to heavily discourage any increase.
        if cost_delta > 0:
            cost_penalty = 1000.0 * (cost_delta ** 2)
        else:
            cost_penalty = 1000.0 * cost_delta  # Linear reward for cost decreases.
        
        reward = -cost_penalty - new_instance_penalty + reuse_bonus
        reward *= self.reward_scale
        return next_state, reward

    def _update_q_table(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
        """
        Update the Q-table using the standard Q-learning update rule.
        """
        old_q = self.q_table.get((state, action), 0.0)
        next_q_values = [self.q_table.get((next_state, a), 0.0)
                         for a in self._all_possible_actions(next_state)]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

    def _all_possible_actions(self, state: Tuple) -> List[Tuple]:
        """
        For multi-step lookahead, generate possible actions based on a dummy task derived from the state.
        """
        dummy_task = {
            'CPU_request': state[3] * 10,
            'memory_request': state[4] * 10,
            'runtime': 1,        # arbitrary runtime
            'timestamp': 0,
            'job_ID': None,
            'task_index': None
        }
        return self._get_possible_actions(dummy_task)

    # -------------------------
    # HELPER METHODS
    # -------------------------
    
    def _calculate_bin_index(self, runtime: float) -> int:
        """
        Bin the runtime similarly to other schedulers.
        """
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1

    def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
        """
        Update the instance's resource usage and cost when assigning a task.
        """
        instance_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
        self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
        self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']
        
        # Update utilization metrics.
        self._update_utilization()
        
        # Price calculation: if this is the first task on the instance.
        if self.instance_bins.at[instance_idx, 'runtime'] == 0:
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = task['runtime']
            self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
        else:
            max_timestamp = max(
                task['runtime'] + task['timestamp'],
                self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
            )
            additional_runtime = (task['runtime'] + task['timestamp']) - (
                self.instance_bins.at[instance_idx, 'runtime'] + self.instance_bins.at[instance_idx, 'timestamp']
            )
            self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[instance_idx, 'runtime'] = max_timestamp - task['timestamp']
            if additional_runtime > 0:
                self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime
        
        # Log the task assignment.
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
        """
        Acquire a new instance of the given type and register it.
        """
        self.instance_counter += 1
        self.instance_id += 1
        instance_type_num = int(instance_type['IndexColumn']) - 1
        new_instance = pd.Series({
            'instance_ID': self.instance_id,
            'bin_index': bin_idx,
            'CPU_capacity': instance_type['capacity_CPU'],
            'CPU_used': 0,
            'memory_capacity': instance_type['capacity_memory'],
            'memory_used': 0,
            'timestamp': 0,
            'runtime': 0,
            'price': instance_type['normalized_price'],
            'instance_type': instance_type_num
        })
        self.instance_counters[instance_type_num] += 1
        self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
        self._update_utilization()
        return new_instance

    def _update_utilization(self):
        """
        Update CPU and memory utilization metrics.
        """
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
        total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()
        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

    def _decay_epsilon(self):
        """
        Gradually reduce epsilon to favor exploitation over exploration.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)









































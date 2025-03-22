# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class DoubleQLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.1,           # Learning rate
#         gamma: float = 0.9,           # Discount factor
#         epsilon: float = 0.3,         # Initial exploration rate
#         epsilon_min: float = 0.01,    # Minimum epsilon
#         epsilon_decay: float = 0.95,  # Decay factor for epsilon per task
#         reward_scale: float = 1.0
#     ):
#         """
#         Double Q-learning–based scheduler that aggressively minimizes cost.
        
#         Key differences from standard Q-learning:
#           - Maintains two Q-tables: Q1 and Q2.
#           - Randomly updates one of them per step, using the other for action selection in the target.
#           - Strong negative rewards for cost increases (quadratic penalty).
#           - Additional penalty for creating new instances.
#           - Small bonus for reusing existing instances.

#         Args:
#             available_instance_types: DataFrame with columns like
#                 ['capacity_CPU','capacity_memory','normalized_price', ...]
#             alpha: Learning rate for Q-updates.
#             gamma: Discount factor for future rewards.
#             epsilon: Probability of random exploration at the start.
#             epsilon_min: Minimum value of epsilon after decay.
#             epsilon_decay: Multiplicative factor for epsilon each step.
#             reward_scale: Scales the final reward value.
#         """

#         # Ensure a contiguous index so enumerating instance_types matches .iloc
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         # Track the maximum CPU/memory across instance types (for possible state binning)
#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks (task_bins) and active instances (instance_bins)
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
#         self.price_counter = 0.0   # Running total cost

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Double Q-learning parameters
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Two Q-tables: Q1 and Q2
#         # Each is a dict mapping (state, action) -> Q-value
#         self.Q1: Dict[Tuple, float] = {}
#         self.Q2: Dict[Tuple, float] = {}

#     # --------------------------------------------------------------------------
#     # Public methods to integrate with your simulation loop
#     # --------------------------------------------------------------------------

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         Updates resource usage and cost accordingly.
#         """
#         # 1. Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             inst_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # 2. Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # 3. Update utilization metrics
#         self._update_utilization()

#     def double_q_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop. For each new task:
#           1. Get current state representation
#           2. Generate possible actions
#           3. Choose action (epsilon-greedy)
#           4. Execute action (assign or create instance)
#           5. Compute reward
#           6. Update either Q1 or Q2 (randomly)
#           7. Decay epsilon
#         """
#         task_list = new_tasks.to_dict('records')
#         for task in task_list:
#             # 1. State representation
#             state = self._get_state_representation(task)

#             # 2. Possible actions
#             actions = self._get_possible_actions(task)

#             # 3. Epsilon-greedy
#             action = self._choose_action(state, actions)

#             # 4. Execute action => next_state, reward
#             next_state, reward = self._execute_action(task, action)

#             # 5. Double Q-learning update
#             self._double_q_update(state, action, reward, next_state)

#             # 6. Decay epsilon
#             self._decay_epsilon()

#     # --------------------------------------------------------------------------
#     # Internal RL logic
#     # --------------------------------------------------------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Builds a coarse state representation with:
#           (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
#         Adjust bin sizes as you see fit.
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         # Discretize CPU/memory usage
#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)

#         num_instances = len(self.instance_bins)

#         # Discretize task request
#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)

#         # Discretize current total cost
#         cost_bin = int(self.price_counter // 50)

#         return (
#             num_instances,
#             cpu_bin,
#             mem_bin,
#             task_cpu_bin,
#             task_mem_bin,
#             cost_bin
#         )

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         For each incoming task, possible actions are:
#           ("use_existing", instance_ID) for each feasible instance
#           ("new_instance", type_index) for each feasible instance type
#           ("unscheduled", None) if no other action is possible
#         """
#         actions = []

#         # 1) Use existing instance if it has enough capacity
#         for _, inst in self.instance_bins.iterrows():
#             if (
#                 inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
#             ):
#                 actions.append(("use_existing", int(inst['instance_ID'])))

#         # 2) Create a new instance from any suitable instance type
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (
#                 inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']
#             ):
#                 actions.append(("new_instance", i))

#         # 3) If no actions, unschedule
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _choose_action(self, state: Tuple, actions: List[Tuple]) -> Tuple:
#         """
#         Epsilon-greedy policy based on combined Q-values: Q(s,a) = (Q1 + Q2)/2.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(actions)

#         # Exploitation: pick argmax of (Q1 + Q2)/2
#         q_values = []
#         for a in actions:
#             q1_val = self.Q1.get((state, a), 0.0)
#             q2_val = self.Q2.get((state, a), 0.0)
#             q_values.append((q1_val + q2_val) / 2.0)

#         best_idx = int(np.argmax(q_values))
#         return actions[best_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Executes the chosen action. Returns (next_state, reward).
#         Reward function strongly penalizes cost increases (quadratic),
#         penalizes creating new instances, and small bonus for reusing existing ones.
#         """
#         action_type, action_val = action

#         old_price = self.price_counter
#         reuse_bonus = 0.0
#         new_inst_penalty = 0.0

#         if action_type == "use_existing":
#             inst_id = action_val
#             instance = self.instance_bins.loc[
#                 self.instance_bins['instance_ID'] == inst_id
#             ].squeeze()
#             reuse_bonus = 50.0  # small reward for reusing instance
#             self._assign_task_to_instance(task, instance)

#         elif action_type == "new_instance":
#             inst_type = self.available_instance_types.iloc[action_val]
#             new_inst_penalty = inst_type['normalized_price'] * 200.0
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)

#         else:
#             # unscheduled
#             new_inst_penalty = 100.0  # treat unscheduling as penalty

#         # Next state
#         next_state = self._get_state_representation(task)

#         # Cost difference
#         new_price = self.price_counter
#         cost_delta = new_price - old_price

#         # Quadratic penalty for cost increases
#         if cost_delta > 0:
#             cost_penalty = 500.0 * (cost_delta ** 2)
#         else:
#             # If cost goes down (rare in this environment, but could happen),
#             # linear "reward" for that. Adjust as needed.
#             cost_penalty = 300.0 * cost_delta

#         # Combine
#         reward = -cost_penalty - new_inst_penalty + reuse_bonus
#         reward *= self.reward_scale

#         return next_state, reward

#     def _double_q_update(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
#         """
#         Double Q-learning update. We randomly pick which table to update:
#           - If we update Q1, we use Q2 to pick the best action in the next state.
#           - If we update Q2, we use Q1 to pick the best action in the next state.
#         """
#         update_q1 = (random.random() < 0.5)  # 50% chance to update Q1

#         # 1. Find next-state best action for the table *not* being updated
#         if update_q1:
#             # We use Q1 to pick best action in s', but evaluate Q2(s', a*) for the target
#             best_a = self._argmax_q_table(self.Q1, next_state)
#             target_q = self.Q2.get((next_state, best_a), 0.0)

#             old_q = self.Q1.get((state, action), 0.0)
#             new_q = old_q + self.alpha * (reward + self.gamma * target_q - old_q)
#             self.Q1[(state, action)] = new_q
#         else:
#             # We use Q2 to pick best action in s', but evaluate Q1(s', a*) for the target
#             best_a = self._argmax_q_table(self.Q2, next_state)
#             target_q = self.Q1.get((next_state, best_a), 0.0)

#             old_q = self.Q2.get((state, action), 0.0)
#             new_q = old_q + self.alpha * (reward + self.gamma * target_q - old_q)
#             self.Q2[(state, action)] = new_q

#     def _argmax_q_table(self, q_table: Dict[Tuple, float], state: Tuple) -> Tuple:
#         """
#         Finds the action with the highest Q-value in the given Q-table for the given state.
#         Because we have a dynamic action space, we must reconstruct possible actions.
#         Then pick the action that yields max Q(s,a).
#         """
#         # We have no direct "next_task" here, so we can't build the next state's actions easily.
#         # A simple approach: we won't do multi-step lookahead for the "best action" in s'.
#         # Instead, we approximate by checking the Q-values for all (s', a) that exist in the table.
#         # Or we can store the last set of possible actions in the table. But let's do a table search.
#         # We'll filter for keys that match 'state' in the first part of the key.

#         # Collect all actions for which (state, action) is in q_table
#         # Then pick the action that has the highest q_table value.
#         relevant_entries = [
#             (sa[1], qval) for sa, qval in q_table.items()
#             if sa[0] == state
#         ]
#         if not relevant_entries:
#             # If no entries found for next_state, return a dummy action
#             return ("unscheduled", None)

#         # relevant_entries is a list of (action, qval)
#         best_action, _ = max(relevant_entries, key=lambda x: x[1])
#         return best_action

#     # --------------------------------------------------------------------------
#     # Additional helper methods
#     # --------------------------------------------------------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Binning for runtime, same as your existing logic."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Updates the instance resource usage, extends cost if needed, logs the task.
#         """
#         idx = self.instance_bins.index[
#             self.instance_bins['instance_ID'] == instance['instance_ID']
#         ][0]

#         self.instance_bins.at[idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[idx, 'memory_used'] += task['memory_request']

#         # Update utilization
#         self._update_utilization()

#         # Cost updates
#         if self.instance_bins.at[idx, 'runtime'] == 0:
#             self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[idx, 'price'] * task['runtime']
#         else:
#             old_end_time = self.instance_bins.at[idx, 'timestamp'] + self.instance_bins.at[idx, 'runtime']
#             new_end_time = task['timestamp'] + task['runtime']
#             max_timestamp = max(old_end_time, new_end_time)

#             additional_runtime = new_end_time - old_end_time
#             self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[idx, 'runtime'] = max_timestamp - task['timestamp']

#             if additional_runtime > 0:
#                 self.price_counter += self.instance_bins.at[idx, 'price'] * additional_runtime

#         # Log the assigned task
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

#     def _acquire_new_instance(self, inst_type: pd.Series, bin_idx: int) -> pd.Series:
#         """
#         Creates a new instance of the given type, tracks it in instance_bins.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1

#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': inst_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': inst_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': inst_type['normalized_price']
#         })

#         self.instance_bins = pd.concat([
#             self.instance_bins, pd.DataFrame([new_instance])
#         ], ignore_index=True)

#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Update CPU and memory utilization based on total capacity vs usage.
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

# class DoubleQLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         alpha: float = 0.05,           # Lower learning rate for stability
#         gamma: float = 0.95,           # Slightly higher discount factor for future cost savings
#         epsilon: float = 0.3,          # Initial exploration rate
#         epsilon_min: float = 0.01,     # Minimum epsilon
#         epsilon_decay: float = 0.99,   # Slower decay for robust exploration early on
#         reward_scale: float = 1.0
#     ):
#         """
#         Double Q-learning–based scheduler that aggressively minimizes cost.
#         This version is modified to focus almost entirely on cost:
#           - Increases in cost (delta > 0) are penalized quadratically with a large factor.
#           - Creating a new instance carries a very high penalty.
#           - Reusing instances yields no bonus (or a negligible one).
        
#         IMPORTANT: To see meaningful improvement, run your entire trace over many episodes
#         (i.e. reset the environment after each run) so the agent can refine its policy.
#         """
#         # Ensure contiguous index for available_instance_types.
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks and active instances.
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Counters.
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0   # Running total cost.

#         # Utilization metrics (kept for state, but not used in reward).
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # Double Q-learning parameters.
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.reward_scale = reward_scale

#         # Two Q-tables.
#         self.Q1: Dict[Tuple, float] = {}
#         self.Q2: Dict[Tuple, float] = {}

#     # --------------------------------------------------------------------------
#     # Public Methods
#     # --------------------------------------------------------------------------

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         Updates resource usage and cost accordingly.
#         """
#         # Free expired tasks.
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             inst_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # Free expired instances.
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # Update utilization metrics.
#         self._update_utilization()

#     def double_q_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop. For each new task:
#           1. Get current state representation.
#           2. Generate possible actions.
#           3. Choose action (epsilon-greedy based on (Q1+Q2)/2).
#           4. Execute action and compute reward.
#           5. Perform Double Q-learning update.
#           6. Decay epsilon.
#         """
#         task_list = new_tasks.to_dict('records')
#         for task in task_list:
#             state = self._get_state_representation(task)
#             actions = self._get_possible_actions(task)
#             action = self._choose_action(state, actions)
#             next_state, reward = self._execute_action(task, action)
#             self._double_q_update(state, action, reward, next_state)
#             self._decay_epsilon()

#     # --------------------------------------------------------------------------
#     # Internal RL Logic
#     # --------------------------------------------------------------------------

#     def _get_state_representation(self, task: Dict) -> Tuple:
#         """
#         Builds a state representation:
#           (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)
#         """
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()

#         cpu_bin = int(total_cpu_used // 50)
#         mem_bin = int(total_mem_used // 50)
#         num_instances = len(self.instance_bins)
#         task_cpu_bin = int(task['CPU_request'] // 10)
#         task_mem_bin = int(task['memory_request'] // 10)
#         cost_bin = int(self.price_counter // 50)

#         return (num_instances, cpu_bin, mem_bin, task_cpu_bin, task_mem_bin, cost_bin)

#     def _get_possible_actions(self, task: Dict) -> List[Tuple]:
#         """
#         Possible actions for the task:
#           ("use_existing", instance_ID) for each instance that can accommodate the task.
#           ("new_instance", type_index) for each instance type that fits the task.
#           ("unscheduled", None) if no action is possible.
#         """
#         actions = []
#         for _, inst in self.instance_bins.iterrows():
#             if (inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
#                 inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']):
#                 actions.append(("use_existing", int(inst['instance_ID'])))
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']):
#                 actions.append(("new_instance", i))
#         if not actions:
#             actions.append(("unscheduled", None))
#         return actions

#     def _choose_action(self, state: Tuple, actions: List[Tuple]) -> Tuple:
#         """
#         Chooses an action using epsilon-greedy based on the combined Q-values (Q1+Q2)/2.
#         """
#         if np.random.rand() < self.epsilon:
#             return random.choice(actions)
#         q_values = []
#         for a in actions:
#             q1_val = self.Q1.get((state, a), 0.0)
#             q2_val = self.Q2.get((state, a), 0.0)
#             q_values.append((q1_val + q2_val) / 2.0)
#         best_idx = int(np.argmax(q_values))
#         return actions[best_idx]

#     def _execute_action(self, task: Dict, action: Tuple) -> Tuple[Tuple, float]:
#         """
#         Executes the chosen action and computes the reward.
#         This reward function is focused on cost minimization:
#           - Any increase in total cost (cost_delta > 0) is penalized quadratically with a high multiplier.
#           - Creating a new instance incurs a very high penalty.
#           - "Unscheduled" tasks are heavily penalized.
#           - Reusing an instance gives no bonus (or a negligible bonus).
#         """
#         action_type, action_val = action
#         old_price = self.price_counter
#         new_inst_penalty = 0.0
#         reuse_bonus = 0.0  # Set to 0 to focus purely on cost.
        
#         if action_type == "use_existing":
#             inst_id = action_val
#             instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id].squeeze()
#             self._assign_task_to_instance(task, instance)
#         elif action_type == "new_instance":
#             inst_type = self.available_instance_types.iloc[action_val]
#             new_inst_penalty = inst_type['normalized_price'] * 1000.0  # Much higher penalty for new instances.
#             new_inst = self._acquire_new_instance(
#                 inst_type,
#                 self._calculate_bin_index(task['runtime'])
#             )
#             self._assign_task_to_instance(task, new_inst)
#         else:
#             new_inst_penalty = 1000.0  # Heavy penalty for unscheduled tasks.

#         next_state = self._get_state_representation(task)
#         new_price = self.price_counter
#         cost_delta = new_price - old_price

#         # Apply a quadratic penalty for cost increases.
#         if cost_delta > 0:
#             cost_penalty = 10000.0 * (cost_delta ** 2)
#         else:
#             cost_penalty = 1000.0 * cost_delta  # Linear reward if cost decreases.
        
#         reward = -cost_penalty - new_inst_penalty + reuse_bonus
#         reward *= self.reward_scale
#         return next_state, reward

#     def _double_q_update(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
#         """
#         Double Q-learning update:
#           - With 50% chance, update Q1 using Q2 for the target, and vice versa.
#         """
#         update_q1 = (random.random() < 0.5)
#         if update_q1:
#             best_a = self._argmax_q_table(self.Q1, next_state)
#             target_q = self.Q2.get((next_state, best_a), 0.0)
#             old_q = self.Q1.get((state, action), 0.0)
#             new_q = old_q + self.alpha * (reward + self.gamma * target_q - old_q)
#             self.Q1[(state, action)] = new_q
#         else:
#             best_a = self._argmax_q_table(self.Q2, next_state)
#             target_q = self.Q1.get((next_state, best_a), 0.0)
#             old_q = self.Q2.get((state, action), 0.0)
#             new_q = old_q + self.alpha * (reward + self.gamma * target_q - old_q)
#             self.Q2[(state, action)] = new_q

#     def _argmax_q_table(self, q_table: Dict[Tuple, float], state: Tuple) -> Tuple:
#         """
#         Returns the action with the highest Q-value for a given state from the provided Q-table.
#         If no entry exists, returns a dummy action.
#         """
#         relevant = [(a, q) for ((s, a), q) in q_table.items() if s == state]
#         if not relevant:
#             return ("unscheduled", None)
#         best_action, _ = max(relevant, key=lambda x: x[1])
#         return best_action

#     # --------------------------------------------------------------------------
#     # Helper Methods
#     # --------------------------------------------------------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """Bins the runtime similarly to your existing logic."""
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Updates instance resource usage and cost, and logs the task assignment.
#         """
#         idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[idx, 'memory_used'] += task['memory_request']
#         self._update_utilization()

#         if self.instance_bins.at[idx, 'runtime'] == 0:
#             self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[idx, 'runtime'] = task['runtime']
#             self.price_counter += self.instance_bins.at[idx, 'price'] * task['runtime']
#         else:
#             old_end = self.instance_bins.at[idx, 'timestamp'] + self.instance_bins.at[idx, 'runtime']
#             new_end = task['timestamp'] + task['runtime']
#             max_time = max(old_end, new_end)
#             additional = new_end - old_end
#             self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
#             self.instance_bins.at[idx, 'runtime'] = max_time - task['timestamp']
#             if additional > 0:
#                 self.price_counter += self.instance_bins.at[idx, 'price'] * additional

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

#     def _acquire_new_instance(self, inst_type: pd.Series, bin_idx: int) -> pd.Series:
#         """
#         Creates and registers a new instance of the given type.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1
#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': inst_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': inst_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': inst_type['normalized_price']
#         })
#         self.instance_bins = pd.concat([
#             self.instance_bins, pd.DataFrame([new_instance])
#         ], ignore_index=True)
#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Updates CPU and memory utilization metrics.
#         """
#         total_cpu_cap = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_mem_cap = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_mem_used = self.instance_bins['memory_used'].sum()
#         self.cpu_utilization = (total_cpu_used / total_cpu_cap) * 100
#         self.memory_utilization = (total_mem_used / total_mem_cap) * 100

#     def _decay_epsilon(self):
#         """
#         Gradually reduces epsilon to shift from exploration to exploitation.
#         """
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# import pandas as pd
# import numpy as np
# from typing import Dict, Tuple, List
# import random

# class DoubleQLearningScheduler:
#     def __init__(
#         self,
#         available_instance_types: pd.DataFrame,
#         population_size: int = 10,
#         num_generations: int = 10,
#         crossover_rate: float = 0.8,
#         mutation_rate: float = 0.1
#     ):
#         """
#         A metaheuristic-based scheduler (using a Genetic Algorithm) that attempts
#         to minimize cost by assigning tasks to either existing instances or newly
#         acquired instances.

#         Args:
#             available_instance_types: DataFrame with columns like
#                 ['capacity_CPU','capacity_memory','normalized_price']
#             population_size: Number of candidate solutions in the GA population
#             num_generations: Number of generations to run the GA
#             crossover_rate: Probability of performing crossover between two solutions
#             mutation_rate: Probability of mutating a solution
#         """
#         # Ensure a contiguous index for instance types
#         self.available_instance_types = available_instance_types.reset_index(drop=True)

#         # Track max CPU/memory for possible binning or checks
#         self.max_cpu = self.available_instance_types['capacity_CPU'].max()
#         self.max_memory = self.available_instance_types['capacity_memory'].max()

#         # DataFrames for tasks and active instances
#         self.task_bins = pd.DataFrame(columns=[
#             'job_ID', 'task_index', 'bin_index', 'instance_ID',
#             'CPU_request', 'memory_request', 'timestamp', 'runtime'
#         ])
#         self.instance_bins = pd.DataFrame(columns=[
#             'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
#             'memory_capacity', 'memory_used', 'timestamp', 'runtime',
#             'price'
#         ])

#         # Scheduler counters
#         self.instance_counter = 0
#         self.instance_id = 0
#         self.tasks = 0
#         self.price_counter = 0.0  # Running total cost

#         # Utilization metrics
#         self.cpu_utilization = 0.0
#         self.memory_utilization = 0.0

#         # GA parameters
#         self.population_size = population_size
#         self.num_generations = num_generations
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate

#     # --------------------------------------------------------------------------
#     # Public Methods (similar to other schedulers)
#     # --------------------------------------------------------------------------

#     def free_tasks_and_instances(self, current_timestamp: float):
#         """
#         Frees expired tasks and instances at the given timestamp.
#         Updates resource usage and cost accordingly.
#         """
#         # 1. Free expired tasks
#         expired_tasks = self.task_bins[
#             self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
#         ]
#         for _, task in expired_tasks.iterrows():
#             inst_id = task['instance_ID']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'CPU_used'] -= task['CPU_request']
#             self.instance_bins.loc[self.instance_bins['instance_ID'] == inst_id, 'memory_used'] -= task['memory_request']
#         self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

#         # 2. Free expired instances
#         expired_instances = self.instance_bins[
#             self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
#         ]
#         self.instance_counter -= len(expired_instances)
#         self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

#         # 3. Update utilization metrics
#         self._update_utilization()

#     def double_q_scheduler(self, new_tasks: pd.DataFrame):
#         """
#         Main scheduling loop for the Metaheuristic (Genetic Algorithm).
#         Steps:
#           1. Convert new_tasks to a list for convenience.
#           2. Run a GA that attempts to find the best assignment for these tasks.
#           3. Apply the best assignment (lowest cost) to update instance_bins, task_bins, etc.
#         """
#         task_list = new_tasks.to_dict('records')
#         if not task_list:
#             return  # No tasks, nothing to schedule

#         # 1. Build an initial population
#         population = self._generate_initial_population(task_list)

#         # 2. GA main loop
#         for _ in range(self.num_generations):
#             # Evaluate fitness (i.e., negative of cost) for each solution
#             fitness_scores = [self._evaluate_solution(solution, task_list) for solution in population]

#             # Select parents
#             parents = self._selection(population, fitness_scores)

#             # Produce offspring
#             offspring = []
#             while len(offspring) < len(population):
#                 p1, p2 = random.sample(parents, 2)
#                 # Crossover
#                 if random.random() < self.crossover_rate:
#                     c1, c2 = self._crossover(p1, p2)
#                 else:
#                     c1, c2 = p1.copy(), p2.copy()
#                 # Mutation
#                 if random.random() < self.mutation_rate:
#                     self._mutate(c1, len(task_list))
#                 if random.random() < self.mutation_rate:
#                     self._mutate(c2, len(task_list))
#                 offspring.append(c1)
#                 offspring.append(c2)

#             # Next generation
#             population = offspring[:len(population)]

#         # 3. Get the best solution from the final population
#         final_fitness_scores = [self._evaluate_solution(sol, task_list) for sol in population]
#         best_idx = int(np.argmax(final_fitness_scores))
#         best_solution = population[best_idx]

#         # 4. Apply the best solution to update our instance_bins and task_bins
#         self._apply_solution(best_solution, task_list)

#     # --------------------------------------------------------------------------
#     # Genetic Algorithm Core
#     # --------------------------------------------------------------------------

#     def _generate_initial_population(self, task_list: List[Dict]) -> List[List[Tuple[str, int]]]:
#         """
#         Generate an initial population of solutions.
#         Each solution is a list of (action_type, action_value) for each task:
#           - ("use_existing", instance_id) or
#           - ("new_instance", instance_type_idx)
#           - or ("unscheduled", None) if no feasible action is found
#         For simplicity, we randomize feasible actions for each task.
#         """
#         population = []
#         for _ in range(self.population_size):
#             solution = []
#             for task in task_list:
#                 feasible_actions = self._get_feasible_actions(task)
#                 if feasible_actions:
#                     solution.append(random.choice(feasible_actions))
#                 else:
#                     solution.append(("unscheduled", None))
#             population.append(solution)
#         return population

#     def _get_feasible_actions(self, task: Dict) -> List[Tuple[str, int]]:
#         """
#         For the sake of building random solutions, we consider:
#           - "new_instance" for any instance_type that can fit the task
#           - "use_existing" is initially empty because we haven't created them yet
#             (We only consider "use_existing" if we want to use existing instance bins, but that requires
#              partial knowledge. For a purely offline approach, we might not do this.)
#         We'll keep it simple: just new_instance for all feasible instance types.
#         """
#         actions = []
#         # Possibly consider "use_existing" for already existing instances if you want partial dynamic logic.
#         # For offline logic, we often treat all tasks from scratch.
#         # We'll skip existing instances for the initial population.

#         # Add "new_instance" for each feasible instance type
#         for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
#             if (inst_type['capacity_CPU'] >= task['CPU_request'] and
#                 inst_type['capacity_memory'] >= task['memory_request']):
#                 actions.append(("new_instance", i))

#         # If no feasible instance type, "unscheduled"
#         if not actions:
#             actions.append(("unscheduled", None))

#         return actions

#     def _selection(self, population: List[List[Tuple]], fitness_scores: List[float]) -> List[List[Tuple]]:
#         """
#         Simple tournament or roulette selection. Here we use a roulette selection approach:
#           - Convert fitness_scores to probabilities
#           - Sample parents from the population with probability proportional to fitness
#         """
#         total_fitness = sum(fitness_scores)
#         if total_fitness <= 0:
#             # if all solutions are invalid or cost is huge, pick randomly
#             return random.sample(population, k=len(population)//2)

#         probs = [f / total_fitness for f in fitness_scores]
#         selected = []
#         for _ in range(len(population)//2):
#             choice_idx = np.random.choice(range(len(population)), p=probs)
#             selected.append(population[choice_idx])
#         return selected

#     def _crossover(self, sol1: List[Tuple[str, int]], sol2: List[Tuple[str, int]]) -> Tuple[List[Tuple], List[Tuple]]:
#         """
#         Single-point crossover between two solutions.
#         """
#         point = random.randint(1, len(sol1) - 1)
#         c1 = sol1[:point] + sol2[point:]
#         c2 = sol2[:point] + sol1[point:]
#         return c1, c2

#     def _mutate(self, solution: List[Tuple[str, int]], num_tasks: int):
#         """
#         Randomly mutate the solution by re-assigning a random index to new_instance or unscheduled, etc.
#         """
#         idx = random.randint(0, num_tasks - 1)
#         # Re-randomize that gene
#         # For simplicity, we just pick a new feasible action for that task
#         # or we can do a small chance to unschedule
#         # We'll skip existing instance logic for brevity
#         # This is a simplified approach
#         solution[idx] = random.choice(self._get_feasible_actions({}))  # empty dict won't find anything though
#         # Actually, let's do something simpler: chance to become "unscheduled"
#         if random.random() < 0.3:
#             solution[idx] = ("unscheduled", None)

#     # --------------------------------------------------------------------------
#     # Evaluate & Apply Solutions
#     # --------------------------------------------------------------------------

#     def _evaluate_solution(self, solution: List[Tuple[str, int]], task_list: List[Dict]) -> float:
#         """
#         For each solution, we simulate the assignment in a "sandbox" environment
#         (cloned instance_bins, etc.), compute the total cost, and return a fitness score
#         (which is negative cost).
#         """
#         # 1. Clone the environment
#         sandbox_instances = pd.DataFrame(columns=self.instance_bins.columns)
#         instance_id_counter = 100000  # large offset to not clash with real environment
#         price_counter = 0.0

#         # 2. For each task in the solution, apply the assigned action
#         for i, (action_type, action_val) in enumerate(solution):
#             task = task_list[i]
#             old_price = price_counter

#             if action_type == "unscheduled":
#                 # big penalty
#                 price_counter += 999999  # effectively kills the solution
#             elif action_type == "new_instance":
#                 # Acquire new instance
#                 inst_type = self.available_instance_types.iloc[action_val]
#                 instance_id_counter += 1
#                 new_inst = pd.Series({
#                     'instance_ID': instance_id_counter,
#                     'bin_index': self._calculate_bin_index(task['runtime']),
#                     'CPU_capacity': inst_type['capacity_CPU'],
#                     'CPU_used': 0,
#                     'memory_capacity': inst_type['capacity_memory'],
#                     'memory_used': 0,
#                     'timestamp': 0,
#                     'runtime': 0,
#                     'price': inst_type['normalized_price']
#                 }, name=len(sandbox_instances))
#                 sandbox_instances = sandbox_instances.append(new_inst)
#                 # Now assign the task
#                 price_counter += self._sandbox_assign_task(task, sandbox_instances, instance_id_counter)
#             else:
#                 # "use_existing", instance_id => we skip in this simplified approach
#                 # but you'd handle existing instance logic if you want partial usage
#                 # We'll treat it like "unscheduled" for now
#                 price_counter += 999999

#             # optionally apply some penalty if cost increased drastically
#             new_price = price_counter
#             cost_delta = new_price - old_price
#             # we won't do a separate penalty here, the final cost is enough

#         # Return negative cost as fitness
#         return -price_counter

#     def _sandbox_assign_task(self, task: Dict, sandbox_instances: pd.DataFrame, instance_id: int) -> float:
#         """
#         Similar logic to `_assign_task_to_instance`, but for the sandbox.
#         Return the cost increase for this assignment.
#         """
#         idx = sandbox_instances.index[sandbox_instances['instance_ID'] == instance_id][0]
#         # Increase resource usage
#         sandbox_instances.at[idx, 'CPU_used'] += task['CPU_request']
#         sandbox_instances.at[idx, 'memory_used'] += task['memory_request']

#         old_end_time = sandbox_instances.at[idx, 'timestamp'] + sandbox_instances.at[idx, 'runtime']
#         new_end_time = task['timestamp'] + task['runtime']
#         additional_runtime = 0.0
#         if new_end_time > old_end_time:
#             additional_runtime = new_end_time - old_end_time

#         sandbox_instances.at[idx, 'timestamp'] = task['timestamp']
#         sandbox_instances.at[idx, 'runtime'] = max(old_end_time, new_end_time) - task['timestamp']

#         cost_increase = 0.0
#         if additional_runtime > 0:
#             cost_increase = sandbox_instances.at[idx, 'price'] * additional_runtime

#         return cost_increase

#     def _apply_solution(self, solution: List[Tuple[str, int]], task_list: List[Dict]):
#         """
#         Applies the best solution to the *real* instance_bins, updates self.price_counter, etc.
#         This modifies the real environment data (unlike the sandbox).
#         """
#         for i, (action_type, action_val) in enumerate(solution):
#             task = task_list[i]
#             if action_type == "unscheduled":
#                 # We'll do nothing, but effectively the task won't be scheduled
#                 # You could apply a penalty or skip
#                 continue
#             elif action_type == "new_instance":
#                 # Acquire a real new instance
#                 inst_type = self.available_instance_types.iloc[action_val]
#                 new_inst = self._acquire_new_instance(inst_type, self._calculate_bin_index(task['runtime']))
#                 self._assign_task_to_instance(task, new_inst)
#             else:
#                 # "use_existing" => in a more advanced approach, you might have a reference
#                 # to an actual instance in instance_bins. For simplicity, we skip here.
#                 # or treat it as unscheduled
#                 pass

#     # --------------------------------------------------------------------------
#     # Helper Methods
#     # --------------------------------------------------------------------------

#     def _calculate_bin_index(self, runtime: float) -> int:
#         """
#         Bins the runtime similarly to your existing logic.
#         """
#         if runtime <= 0:
#             return 0
#         return int(np.floor(np.log2(runtime))) + 1

#     def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
#         """
#         Updates instance resource usage and cost in the real environment,
#         logs the task assignment in self.task_bins.
#         """
#         idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']][0]
#         self.instance_bins.at[idx, 'CPU_used'] += task['CPU_request']
#         self.instance_bins.at[idx, 'memory_used'] += task['memory_request']

#         # Update utilization
#         self._update_utilization()

#         old_end_time = self.instance_bins.at[idx, 'timestamp'] + self.instance_bins.at[idx, 'runtime']
#         new_end_time = task['timestamp'] + task['runtime']
#         additional_runtime = 0.0
#         if new_end_time > old_end_time:
#             additional_runtime = new_end_time - old_end_time

#         self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
#         self.instance_bins.at[idx, 'runtime'] = max(old_end_time, new_end_time) - task['timestamp']

#         if additional_runtime > 0:
#             self.price_counter += self.instance_bins.at[idx, 'price'] * additional_runtime

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

#     def _acquire_new_instance(self, inst_type: pd.Series, bin_idx: int) -> pd.Series:
#         """
#         Creates and registers a new instance in the real environment.
#         """
#         self.instance_counter += 1
#         self.instance_id += 1
#         new_instance = pd.Series({
#             'instance_ID': self.instance_id,
#             'bin_index': bin_idx,
#             'CPU_capacity': inst_type['capacity_CPU'],
#             'CPU_used': 0,
#             'memory_capacity': inst_type['capacity_memory'],
#             'memory_used': 0,
#             'timestamp': 0,
#             'runtime': 0,
#             'price': inst_type['normalized_price']
#         })
#         self.instance_bins = pd.concat([
#             self.instance_bins, pd.DataFrame([new_instance])
#         ], ignore_index=True)
#         self._update_utilization()
#         return new_instance

#     def _update_utilization(self):
#         """
#         Updates CPU and memory utilization metrics in the real environment.
#         """
#         total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
#         total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
#         total_cpu_used = self.instance_bins['CPU_used'].sum()
#         total_memory_used = self.instance_bins['memory_used'].sum()

#         self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
#         self.memory_utilization = (total_memory_used / total_memory_capacity) * 100

# 

import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        """
        Prioritized replay buffer.
        Args:
            capacity: maximum number of transitions.
            alpha: exponent that determines how much prioritization is used (0 = uniform sampling).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        # When adding a new transition, assign it the maximum current priority.
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        # Use all current priorities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling (IS) weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability

        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones), weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.priorities = []
        self.pos = 0

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DuelingDQN, self).__init__()
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage into Q-values.
        # Subtract mean advantage for stability.
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

class DeepQLearningScheduler:
    def __init__(
        self,
        available_instance_types: pd.DataFrame,
        # Neural network & training hyperparameters
        state_dim: int = 10,          # Enhanced state representation
        hidden_dim: int = 128,        # Larger hidden layer
        lr: float = 5e-4,             # Adjusted learning rate
        gamma: float = 0.99,          # Higher discount factor for long-term rewards
        epsilon: float = 1.0,         # Initial epsilon for exploration
        epsilon_min: float = 0.05,    # Slightly higher minimum epsilon
        epsilon_decay: float = 0.997, # Slower decay for better exploration
        replay_size: int = 50000,     # Larger replay buffer
        batch_size: int = 128,        # Larger batch size
        target_update_freq: int = 100,# More frequent target updates
        # Reward function constants
        new_instance_penalty: float = 10.0,  # Base penalty for new instances
        instance_utilization_reward: float = 100.0,  # Reward for good utilization
        cost_efficiency_reward: float = 1000.0,  # Reward for cost efficiency
        waste_penalty: float = 500.0,  # Penalty for wasted resources
        # Priority experience replay
        use_prioritized_replay: bool = True,
        alpha: float = 0.6,           # Priority exponent (0 = uniform, 1 = greedy)
        beta: float = 0.4,            # Importance sampling exponent
        beta_increment: float = 0.001, # Beta increment per step
        # Pretraining and other settings
        enable_lookahead: bool = True,
        lookahead_window: int = 5,
        min_replay_before_train: int = 1000,
        reward_scale: float = 1.0  # Default value
    ):
        """
        Improved Deep Q-learning scheduler with enhanced state representation,
        prioritized experience replay, and future task consideration.
        
        Args:
            available_instance_types: DataFrame with columns:
                ['capacity_CPU','capacity_memory','normalized_price']
            state_dim: Dimension of the state vector.
            hidden_dim: Size of hidden layers in the neural network.
            lr: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Rate of epsilon decay.
            replay_size: Size of the replay buffer.
            batch_size: Batch size for training.
            target_update_freq: Frequency of target network updates.
            new_instance_penalty: Base penalty for creating new instances.
            instance_utilization_reward: Reward factor for good utilization.
            cost_efficiency_reward: Reward factor for cost efficiency.
            waste_penalty: Penalty factor for wasted resources.
            use_prioritized_replay: Whether to use prioritized experience replay.
            alpha: Priority exponent.
            beta: Importance sampling exponent.
            beta_increment: Increment of beta per step.
            enable_lookahead: Whether to consider future tasks.
            lookahead_window: Number of future tasks to consider.
            min_replay_before_train: Minimum replay buffer size before training.
        """
        # Reset and initialize instance types
        self.available_instance_types = available_instance_types.reset_index(drop=True)
        self.max_cpu = self.available_instance_types['capacity_CPU'].max()
        self.max_memory = self.available_instance_types['capacity_memory'].max()
        self.min_cpu = self.available_instance_types['capacity_CPU'].min()
        self.min_memory = self.available_instance_types['capacity_memory'].min()
        
        # Sort instance types by price-to-resource ratio (most efficient first)
        self.available_instance_types['cpu_mem_total'] = (
            self.available_instance_types['capacity_CPU'] + 
            self.available_instance_types['capacity_memory']
        )
        self.available_instance_types['price_efficiency'] = (
            self.available_instance_types['normalized_price'] / 
            self.available_instance_types['cpu_mem_total']
        )
        self.available_instance_types = self.available_instance_types.sort_values(
            'price_efficiency', ascending=True
        ).reset_index(drop=True)

        # Initialize dataframes for tracking
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 'task_index', 'bin_index', 'instance_ID',
            'CPU_request', 'memory_request', 'timestamp', 'runtime'
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
            'price',
            'instance_type_index'
        ])

        # Scheduler counters
        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.price_counter = 0.0
        self.total_reward = 0.0
        self.current_timestamp = 0.0

        # Utilization metrics
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.cost_efficiency = 0.0

        # DQL parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        self.total_steps = 0

               # Reward function parameters
        self.new_instance_penalty = new_instance_penalty
        self.instance_utilization_reward = instance_utilization_reward
        self.cost_efficiency_reward = cost_efficiency_reward
        self.waste_penalty = waste_penalty
        self.reward_scale = reward_scale  # This line sets the attribute.
        
        # Add cost penalty factors if not provided
        self.cost_penalty_factor_quad = 10000.0
        self.cost_penalty_factor_linear = 1000.0


        # Additional settings
        self.enable_lookahead = enable_lookahead
        self.lookahead_window = lookahead_window
        self.future_tasks_buffer = deque(maxlen=lookahead_window)
        self.min_replay_before_train = min_replay_before_train

        # Replay buffer
        self.replay_size = replay_size
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(replay_size, alpha)
            self.beta = beta
            self.beta_increment = beta_increment
        else:
            self.replay_buffer = ReplayBuffer(replay_size)

        # Network dimensions
        self.state_dim = state_dim
        self.action_dim = len(self.available_instance_types) + 1  # +1 for "no action"
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingDQN(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------
    def free_tasks_and_instances(self, current_timestamp: float):
        """
        Remove expired tasks and instances based on current timestamp.
        Update resource utilization and metrics.
        """
        self.current_timestamp = current_timestamp
        expired_tasks = self.task_bins[
            self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
        ]
        for _, task in expired_tasks.iterrows():
            inst_id = task['instance_ID']
            inst_idx = self.instance_bins.index[self.instance_bins['instance_ID'] == inst_id]
            if len(inst_idx) > 0:
                self.instance_bins.loc[inst_idx, 'CPU_used'] -= task['CPU_request']
                self.instance_bins.loc[inst_idx, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]
        expired_instances = self.instance_bins[
            self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
        ]
        self.instance_counter -= len(expired_instances)
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]
        # Clean up instances with no tasks
        empty_instances = self.instance_bins[
            (self.instance_bins['CPU_used'] == 0) & 
            (self.instance_bins['memory_used'] == 0)
        ]
        self.instance_counter -= len(empty_instances)
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(empty_instances.index)]
        self._update_utilization()
        self.instance_bins['CPU_used'] = self.instance_bins['CPU_used'].clip(lower=0)
        self.instance_bins['memory_used'] = self.instance_bins['memory_used'].clip(lower=0)

    def add_future_tasks(self, future_tasks: pd.DataFrame):
        """
        Store future tasks for lookahead scheduling.
        """
        if self.enable_lookahead:
            for _, task in future_tasks.iterrows():
                self.future_tasks_buffer.append(task)

    def schedule_task(self, task: Dict) -> Optional[int]:
        """
        Schedule a single task using the DQL algorithm.
        Returns the scheduled instance ID.
        """
        state = self._get_state_representation(task)
        possible_actions = self._get_possible_actions(task)
        if not possible_actions:
            return None
        action = self._choose_action(state, possible_actions)
        next_state, reward, instance_id = self._execute_action(task, action)
        self._store_transition(state, action, reward, next_state, False)
        self.total_reward += reward
        if self.total_steps > self.min_replay_before_train:
            self._train_step()
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        self._decay_epsilon()
        self.total_steps += 1
        return instance_id

    def deep_q_scheduler(self, new_tasks: pd.DataFrame):
        """
        Schedule multiple tasks using the DQL algorithm.
        """
        task_list = new_tasks.to_dict('records')
        for task in task_list:
            self.schedule_task(task)

    def reset_environment(self):
        """
        Reset the scheduling environment for a new episode.
        """
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 'task_index', 'bin_index', 'instance_ID',
            'CPU_request', 'memory_request', 'timestamp', 'runtime'
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
            'price',
            'instance_type_index'
        ])
        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.price_counter = 0.0
        self.total_reward = 0.0
        self.current_timestamp = 0.0
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.cost_efficiency = 0.0
        self.future_tasks_buffer.clear()
        # Optionally clear replay buffer if desired:
        if self.use_prioritized_replay:
            self.replay_buffer.reset()  # Assuming your prioritized buffer has a reset method.
        else:
            self.replay_buffer.clear()
        # Reset epsilon for new episode.
        self.epsilon = 1.0

    # --------------------------------------------------------------------------
    # Internal Deep Q-learning Logic
    # --------------------------------------------------------------------------
    def _get_state_representation(self, task: Dict) -> np.ndarray:
        """
        Create a richer state representation with normalized values.
        """
        # Instance-level features
        num_instances = len(self.instance_bins)
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() if num_instances > 0 else 0
        total_cpu_used = self.instance_bins['CPU_used'].sum() if num_instances > 0 else 0
        total_mem_capacity = self.instance_bins['memory_capacity'].sum() if num_instances > 0 else 0
        total_mem_used = self.instance_bins['memory_used'].sum() if num_instances > 0 else 0
        
        # Task features
        task_cpu = task['CPU_request']
        task_mem = task['memory_request']
        task_runtime = task['runtime']
        
        # Normalize values
        norm_num_instances = num_instances / 50  # Assuming max 50 instances
        norm_cpu_util = total_cpu_used / max(total_cpu_capacity, 1)
        norm_mem_util = total_mem_used / max(total_mem_capacity, 1)
        norm_task_cpu = task_cpu / self.max_cpu
        norm_task_mem = task_mem / self.max_memory
        norm_task_runtime = np.log1p(task_runtime) / 10  # Log-scale normalization
        
        # Calculate best-fit metrics
        best_fit_cpu = 0
        best_fit_mem = 0
        avail_cpu = []
        avail_mem = []
        for _, inst in self.instance_bins.iterrows():
            avail_cpu.append(inst['CPU_capacity'] - inst['CPU_used'])
            avail_mem.append(inst['memory_capacity'] - inst['memory_used'])
        if avail_cpu:
            valid = [i for i in range(len(avail_cpu)) if avail_cpu[i] >= task_cpu and avail_mem[i] >= task_mem]
            if valid:
                best_idx = min(valid, key=lambda i: (avail_cpu[i] - task_cpu) + (avail_mem[i] - task_mem))
                best_fit_cpu = avail_cpu[best_idx] / self.max_cpu
                best_fit_mem = avail_mem[best_idx] / self.max_memory
        
        # Lookahead features
        future_task_cpu = 0
        future_task_mem = 0
        if self.enable_lookahead and self.future_tasks_buffer:
            future_tasks = list(self.future_tasks_buffer)
            future_task_cpu = sum(t['CPU_request'] for t in future_tasks) / (self.max_cpu * len(future_tasks))
            future_task_mem = sum(t['memory_request'] for t in future_tasks) / (self.max_memory * len(future_tasks))
        
        state = np.array([
            norm_num_instances,
            norm_cpu_util,
            norm_mem_util,
            norm_task_cpu,
            norm_task_mem,
            norm_task_runtime,
            best_fit_cpu,
            best_fit_mem,
            future_task_cpu,
            future_task_mem
        ], dtype=np.float32)
        return state

    def _get_possible_actions(self, task: Dict) -> List[Dict]:
        """
        Generate possible actions for a task.
        Each action is a dictionary with details.
        """
        actions = []
        # Option 1: Place on existing instance.
        for idx, inst in self.instance_bins.iterrows():
            if (inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
                inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']):
                actions.append({
                    'type': 'use_existing',
                    'instance_id': int(inst['instance_ID']),
                    'idx': idx,
                    'waste': ((inst['CPU_capacity'] - inst['CPU_used'] - task['CPU_request']) +
                              (inst['memory_capacity'] - inst['memory_used'] - task['memory_request'])) /
                              (inst['CPU_capacity'] + inst['memory_capacity'])
                })
        # Option 2: Create new instance.
        for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
            if (inst_type['capacity_CPU'] >= task['CPU_request'] and
                inst_type['capacity_memory'] >= task['memory_request']):
                actions.append({
                    'type': 'new_instance',
                    'instance_type_idx': i,
                    'waste': ((inst_type['capacity_CPU'] - task['CPU_request']) +
                              (inst_type['capacity_memory'] - task['memory_request'])) /
                              (inst_type['capacity_CPU'] + inst_type['capacity_memory'])
                })
        # Sort by waste (lowest first)
        actions.sort(key=lambda x: x['waste'])
        return actions

    def _choose_action(self, state: np.ndarray, possible_actions: List[Dict]) -> Dict:
        """
        Choose an action using an epsilon-greedy strategy.
        """
        if not possible_actions:
            return None
        if random.random() < self.epsilon:
            return possible_actions[0]  # Prefer the action with lowest waste during exploration.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_tensor).cpu().numpy().flatten()
        # Map Q-values to possible actions.
        action_values = []
        for action in possible_actions:
            if action['type'] == 'use_existing':
                # Map all use_existing actions to index 0.
                action_values.append((action, q_vals[0]))
            else:
                action_idx = action['instance_type_idx'] + 1
                action_values.append((action, q_vals[action_idx]))
        best_action, _ = max(action_values, key=lambda x: x[1])
        return best_action

    def _execute_action(self, task: Dict, action: Dict) -> Tuple[np.ndarray, float, int]:
        # Directly get action type from the dictionary.
        action_type = action['type']
        old_price = self.price_counter
        new_inst_penalty = 0.0
        instance_id = None

        if action_type == 'use_existing':
            instance_id = action['instance_id']
            # Use instance_id to find the instance
            instance = self.instance_bins[self.instance_bins['instance_ID'] == instance_id].iloc[0]
            self._assign_task_to_instance(task, instance)
        elif action_type == 'new_instance':
            inst_type = self.available_instance_types.iloc[action['instance_type_idx']]
            new_inst_penalty = inst_type['normalized_price'] * self.new_instance_penalty
            new_inst = self._acquire_new_instance(inst_type, self._calculate_bin_index(task['runtime']),
                                                action['instance_type_idx'])
            instance_id = new_inst['instance_ID']
            self._assign_task_to_instance(task, new_inst)
        else:
            # Fallback (should not happen if we force scheduling)
            new_inst_penalty = 1000.0

        new_price = self.price_counter
        cost_delta = new_price - old_price

        if cost_delta > 0:
            cost_penalty = self.cost_penalty_factor_quad * (cost_delta ** 2)
        else:
            cost_penalty = self.cost_penalty_factor_linear * cost_delta

        reward = -cost_penalty - new_inst_penalty
        reward *= self.reward_scale

        next_state = self._get_state_representation(task)
        return next_state, reward, instance_id



    def _store_transition(self, state: np.ndarray, action: Dict, reward: float, next_state: np.ndarray, done: bool):
        """
        Store transition in the replay buffer.
        Convert action to a numeric index.
        """
        if action['type'] == 'use_existing':
            action_idx = 0
        else:
            action_idx = action['instance_type_idx'] + 1
        self.replay_buffer.add(state, action_idx, reward, next_state, done)

    def _train_step(self):
        """
        Train the Q-network on a batch of transitions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.use_prioritized_replay:
            batch, weights, indices = self.replay_buffer.sample(self.batch_size, self.beta)
            states, actions, rewards, next_states, dones = batch
            states_t = torch.FloatTensor(states).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            dones_t = torch.BoolTensor(dones).to(self.device)
            weights_t = torch.FloatTensor(weights).to(self.device)
            
            current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_net(next_states_t).max(1)[1]
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_t + (1 - dones_t.float()) * self.gamma * next_q
            
            td_errors = target_q - current_q
            loss = (weights_t * td_errors.pow(2)).mean()
            self.replay_buffer.update_priorities(indices, np.abs(td_errors.cpu().detach().numpy()) + 1e-6)
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            states_t = torch.FloatTensor(states).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)
            
            current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_net(next_states_t).max(1)[1]
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
            
            loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        self.scheduler.step()
        self.learn_step_counter += 1

    def _update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    def _calculate_bin_index(self, runtime: float) -> int:
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1

    def _assign_task_to_instance(self, task: Dict, instance: pd.Series):
        idx = self.instance_bins.index[self.instance_bins['instance_ID'] == instance['instance_ID']].tolist()[0]
        self.instance_bins.at[idx, 'CPU_used'] += task['CPU_request']
        self.instance_bins.at[idx, 'memory_used'] += task['memory_request']
        self._update_utilization()

        if self.instance_bins.at[idx, 'runtime'] == 0: 
            self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[idx, 'runtime'] = task['runtime']
            self.price_counter += self.instance_bins.at[idx, 'price'] * task['runtime']
        else:
            max_timestamp = max(
                task['runtime'] + task['timestamp'], 
                self.instance_bins.at[idx, 'runtime'] + self.instance_bins.at[idx, 'timestamp']
            )
            additional_runtime = task['runtime'] + task['timestamp'] - (self.instance_bins.at[idx, 'runtime'] + self.instance_bins.at[idx, 'timestamp'])
            self.instance_bins.at[idx, 'timestamp'] = task['timestamp']
            self.instance_bins.at[idx, 'runtime'] = max_timestamp - task['timestamp']
            if additional_runtime > 0:
                self.price_counter += self.instance_bins.at[idx, 'price'] * additional_runtime


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
        self._update_utilization()

    def _acquire_new_instance(self, inst_type: pd.Series, bin_idx: int, type_idx: int) -> pd.Series:
        self.instance_counter += 1
        self.instance_id += 1
        new_instance = pd.Series({
            'instance_ID': self.instance_id,
            'bin_index': bin_idx,
            'CPU_capacity': inst_type['capacity_CPU'],
            'CPU_used': 0,
            'memory_capacity': inst_type['capacity_memory'],
            'memory_used': 0,
            'timestamp': 0,
            'runtime': 0,
            'price': inst_type['normalized_price'],
            'instance_type_index': type_idx
        })
        self.instance_bins = pd.concat([self.instance_bins, pd.DataFrame([new_instance])], ignore_index=True)
        self._update_utilization()
        return new_instance

    def _update_utilization(self):
        """
        Update resource utilization metrics.
        """
        if len(self.instance_bins) == 0:
            self.cpu_utilization = 0.0
            self.memory_utilization = 0.0
        else:
            total_cpu_cap = self.instance_bins['CPU_capacity'].sum() or 1e-9
            total_mem_cap = self.instance_bins['memory_capacity'].sum() or 1e-9
            total_cpu_used = self.instance_bins['CPU_used'].sum()
            total_mem_used = self.instance_bins['memory_used'].sum()
            self.cpu_utilization = (total_cpu_used / total_cpu_cap) * 100
            self.memory_utilization = (total_mem_used / total_mem_cap) * 100

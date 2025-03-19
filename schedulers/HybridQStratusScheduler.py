import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import random

class HybridQStratusScheduler:
    def __init__(
        self,
        available_instance_types: pd.DataFrame,
        alpha: float = 0.1,           # Learning rate
        gamma: float = 0.9,           # Discount factor
        epsilon: float = 0.3,         # Starting epsilon for exploration
        epsilon_min: float = 0.01,    # Minimum epsilon after decay
        epsilon_decay: float = 0.95,  # Decay rate
        reward_scale: float = 1.0,
        q_influence: float = 0.6      # Weight given to Q-learning decisions (0.0-1.0)
    ):
        """
        A hybrid scheduler that combines Q-learning and Stratus bin-packing approaches.

        Args:
            available_instance_types: DataFrame with columns ['capacity_CPU', 'capacity_memory', 'normalized_price']
            alpha: Learning rate for Q-learning
            gamma: Discount factor for Q-learning
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            reward_scale: Scaling factor for rewards
            q_influence: Weight given to Q-learning decisions vs Stratus (0.0-1.0)
        """
        # Reset index for consistent iloc references
        self.available_instance_types = available_instance_types.reset_index(drop=True)
        self.max_cpu = self.available_instance_types['capacity_CPU'].max()
        self.max_memory = self.available_instance_types['capacity_memory'].max()

        # DataFrames for tasks and instances
        self.task_bins = pd.DataFrame(columns=[
            'job_ID', 'task_index', 'bin_index', 'instance_ID',
            'CPU_request', 'memory_request', 'timestamp', 'runtime'
        ])
        self.instance_bins = pd.DataFrame(columns=[
            'instance_ID', 'bin_index', 'CPU_capacity', 'CPU_used',
            'memory_capacity', 'memory_used', 'timestamp', 'runtime',
            'price','instance_type'
        ])

        # Counters and metrics
        self.instance_counter = 0
        self.instance_id = 0
        self.tasks = 0
        self.price_counter = 0.0
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0

        self.instance_counters = [0] * 10


        # Q-learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale
        self.q_influence = q_influence

        # Q-table: mapping (state, action) -> Q-value
        self.q_table: Dict[Tuple, float] = {}

        # Performance tracking
        self.decisions = {"q_learning": 0, "stratus": 0, "hybrid": 0}
        self.running_cost_savings = 0.0

    # --------------------------------------------------------------------------
    # Cleanup for expired tasks/instances
    # --------------------------------------------------------------------------
    def free_tasks_and_instances(self, current_timestamp: float):
        """
        Frees expired tasks and instances at the given timestamp.
        """
        # Free tasks that have expired
        expired_tasks = self.task_bins[
            self.task_bins['timestamp'] + self.task_bins['runtime'] <= current_timestamp
        ]
        for _, task in expired_tasks.iterrows():
            instance_id = task['instance_ID']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'CPU_used'] -= task['CPU_request']
            self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id, 'memory_used'] -= task['memory_request']
        self.task_bins = self.task_bins[~self.task_bins.index.isin(expired_tasks.index)]

        # Free expired instances
        expired_instances = self.instance_bins[
            self.instance_bins['timestamp'] + self.instance_bins['runtime'] <= current_timestamp
        ]
        for _, instance in expired_instances.iterrows():
            instance_type_index = int(instance['instance_type'])
            self.instance_counters[instance_type_index] -= 1
        self.instance_counter -= len(expired_instances)
        self.instance_bins = self.instance_bins[~self.instance_bins.index.isin(expired_instances.index)]

        # Update utilization metrics
        self._update_utilization()

    # --------------------------------------------------------------------------
    # Main scheduling method
    # --------------------------------------------------------------------------
    def packer(self, new_tasks: pd.DataFrame):
        """
        Main scheduling method that combines Q-learning and Stratus approaches.
        First attempts a hybrid Q-learning + Stratus decision for each task,
        then uses a Stratus-based scaler to handle leftover tasks.
        """
        # Sort tasks by runtime descending (Stratus approach)
        sorted_tasks = new_tasks.sort_values('runtime', ascending=False)
        unscheduled_tasks = []

        for _, task in sorted_tasks.iterrows():
            # Build the state representation
            state = self._get_state_representation(task)

            # Get possible actions
            possible_actions = self._get_possible_actions(task)
            if not possible_actions or all(a[0] == "unscheduled" for a in possible_actions):
                # If no valid actions, mark as unscheduled
                unscheduled_tasks.append(task)
                continue

            # Possibly use Q-learning to pick an action
            q_decision = None
            if len(self.q_table) > 10 and random.random() < self.q_influence:
                # If we have a sufficiently populated Q-table and random check passes,
                # use Q-learning to decide
                q_decision = self._choose_q_action(state, possible_actions)

            # Otherwise, get Stratus's action
            stratus_decision = self._choose_stratus_action(task, possible_actions)

            # Merge decisions (hybrid)
            final_decision = self._make_hybrid_decision(q_decision, stratus_decision, state)

            # Execute final decision
            old_price = self.price_counter
            if final_decision[0] == "use_existing":
                instance_id = final_decision[1]
                instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
                self._assign_task_to_instance(task, instance)
            elif final_decision[0] == "new_instance":
                instance_type_idx = final_decision[1]
                inst_type = self.available_instance_types.iloc[instance_type_idx]
                new_inst = self._acquire_new_instance(
                    inst_type,
                    self._calculate_bin_index(task['runtime'])
                )
                self._assign_task_to_instance(task, new_inst)
            else:  # "unscheduled"
                unscheduled_tasks.append(task)

            # Compute reward and update Q-table
            new_price = self.price_counter
            reward = self._calculate_reward(old_price, new_price, final_decision)
            next_state = self._get_state_representation(task)
            self._update_q_table(state, final_decision, reward, next_state)

            # Decay epsilon
            self._decay_epsilon()

        # Scale out leftover tasks using Stratus logic
        self._stratus_scaler(unscheduled_tasks)

    # --------------------------------------------------------------------------
    # Q-Learning Components
    # --------------------------------------------------------------------------
    def _get_state_representation(self, task: Dict) -> Tuple:
        """
        Creates a state representation that incorporates both Q-learning
        and Stratus-like metrics.
        """
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_mem_used = self.instance_bins['memory_used'].sum()

        cpu_bin = int(total_cpu_used // 50)
        mem_bin = int(total_mem_used // 50)
        num_instances = len(self.instance_bins)

        task_cpu_bin = int(task['CPU_request'] // 10)
        task_mem_bin = int(task['memory_request'] // 10)
        task_runtime_bin = self._calculate_bin_index(task['runtime'])

        # Discretize current cost
        cost_bin = int(self.price_counter // 50)

        # Include utilization metrics
        cpu_util_bin = int(self.cpu_utilization // 10)
        mem_util_bin = int(self.memory_utilization // 10)

        state = (
            num_instances,
            cpu_bin,
            mem_bin,
            task_cpu_bin,
            task_mem_bin,
            task_runtime_bin,
            cost_bin,
            cpu_util_bin,
            mem_util_bin
        )
        return state

    def _get_possible_actions(self, task: Dict) -> List[Tuple]:
        """
        Gets all possible actions for a task: use existing instances, create new ones,
        or leave unscheduled.
        """
        actions = []

        # 1) Use existing instances
        for _, inst in self.instance_bins.iterrows():
            if (
                inst['CPU_capacity'] - inst['CPU_used'] >= task['CPU_request'] and
                inst['memory_capacity'] - inst['memory_used'] >= task['memory_request']
            ):
                actions.append(("use_existing", int(inst['instance_ID'])))

        # 2) Acquire new instance
        for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
            if (
                inst_type['capacity_CPU'] >= task['CPU_request'] and
                inst_type['capacity_memory'] >= task['memory_request']
            ):
                actions.append(("new_instance", i))

        # 3) Fallback: unscheduled
        if not actions:
            actions.append(("unscheduled", None))

        return actions

    def _choose_q_action(self, state: Tuple, possible_actions: List[Tuple]) -> Tuple:
        """
        Epsilon-greedy Q-learning decision.
        """
        if np.random.rand() < self.epsilon:
            return random.choice(possible_actions)

        q_values = [self.q_table.get((state, action), 0.0) for action in possible_actions]
        best_idx = int(np.argmax(q_values))
        return possible_actions[best_idx]

    def _update_q_table(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple):
        """
        Standard Q-learning update.
        """
        old_q = self.q_table.get((state, action), 0.0)
        next_actions = self._get_all_possible_actions(next_state)
        next_q_values = [self.q_table.get((next_state, a), 0.0) for a in next_actions]
        max_next_q = max(next_q_values) if next_q_values else 0.0

        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

    def _calculate_reward(self, old_price: float, new_price: float, action: Tuple) -> float:
        """
        Calculate reward based on cost changes, action type, and utilization improvements.
        """
        action_type, action_value = action
        cost_delta = new_price - old_price

        # Cost penalty or reward
        if cost_delta > 0:
            # Quadratic penalty for cost increase
            cost_penalty = 500.0 * (cost_delta ** 2)
        else:
            # Mild reward if cost goes down or stays same
            cost_penalty = 500.0 * cost_delta

        # Action-specific adjustments
        action_bonus = 0.0
        if action_type == "use_existing":
            action_bonus = 100.0  # Bonus for reusing an existing instance
        elif action_type == "new_instance":
            if action_value is not None:
                # Heavier penalty for more expensive instance types
                action_bonus = -100.0 * self.available_instance_types.iloc[action_value]['normalized_price']
        else:  # "unscheduled"
            action_bonus = -50.0  # Penalty for leaving task unscheduled

        # Utilization bonus
        utilization_bonus = (self.cpu_utilization + self.memory_utilization) / 4.0

        reward = -cost_penalty + action_bonus + utilization_bonus
        return reward * self.reward_scale

    def _decay_epsilon(self):
        """
        Gradually reduce epsilon to favor exploitation over exploration.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _get_all_possible_actions(self, state: Tuple) -> List[Tuple]:
        """
        Helper method to find all known actions for the given state in the Q-table.
        """
        return [action for (s, action) in self.q_table.keys() if s == state]

    # --------------------------------------------------------------------------
    # Stratus-like Decision Logic
    # --------------------------------------------------------------------------
    def _choose_stratus_action(self, task: Dict, possible_actions: List[Tuple]) -> Tuple:
        """
        Choose an action based on Stratus bin-packing principles:
          1) Try same bin
          2) Try higher bins
          3) Try lower bins
          4) If none, pick the cheapest new instance
        """
        task_bin = self._calculate_bin_index(task['runtime'])

        # Separate actions
        existing_actions = [a for a in possible_actions if a[0] == "use_existing"]
        new_instance_actions = [a for a in possible_actions if a[0] == "new_instance"]

        # 1) Check same bin
        same_bin_list = []
        for action in existing_actions:
            instance_id = action[1]
            instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
            if instance['bin_index'] == task_bin:
                same_bin_list.append((action, instance))

        if same_bin_list:
            # pick instance with the smallest runtime difference
            runtime_diffs = [abs(inst['runtime'] - task['runtime']) for (_, inst) in same_bin_list]
            best_idx = int(np.argmin(runtime_diffs))
            return same_bin_list[best_idx][0]

        # 2) Check higher bins
        higher_bin_list = []
        for action in existing_actions:
            instance_id = action[1]
            instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
            if instance['bin_index'] > task_bin:
                higher_bin_list.append((action, instance))

        if higher_bin_list:
            # pick instance with the most available resources
            resources = [
                (inst['CPU_capacity'] - inst['CPU_used']) + (inst['memory_capacity'] - inst['memory_used'])
                for (_, inst) in higher_bin_list
            ]
            best_idx = int(np.argmax(resources))
            return higher_bin_list[best_idx][0]

        # 3) Check lower bins
        lower_bin_list = []
        for action in existing_actions:
            instance_id = action[1]
            instance = self.instance_bins.loc[self.instance_bins['instance_ID'] == instance_id].squeeze()
            if instance['bin_index'] < task_bin:
                lower_bin_list.append((action, instance))

        if lower_bin_list:
            # pick instance with the most available resources
            resources = [
                (inst['CPU_capacity'] - inst['CPU_used']) + (inst['memory_capacity'] - inst['memory_used'])
                for (_, inst) in lower_bin_list
            ]
            best_idx = int(np.argmax(resources))
            return lower_bin_list[best_idx][0]

        # 4) No existing instance is suitable, pick the cheapest new instance
        if new_instance_actions:
            cheapest_price = float('inf')
            best_action = None
            for (act_type, idx) in new_instance_actions:
                price = self.available_instance_types.iloc[idx]['normalized_price']
                if price < cheapest_price:
                    cheapest_price = price
                    best_action = (act_type, idx)
            return best_action

        # If none, fallback to unscheduled
        return ("unscheduled", None)

    def _make_hybrid_decision(self, q_decision: Tuple, stratus_decision: Tuple, state: Tuple) -> Tuple:
        """
        Merge the Q-learning decision with the Stratus decision. If Q-learning is available,
        we can pick it. Otherwise, fallback to Stratus.
        """
        if q_decision is not None:
            self.decisions["q_learning"] += 1
            return q_decision
        else:
            self.decisions["stratus"] += 1
            return stratus_decision

    # --------------------------------------------------------------------------
    # Stratus-Style Scale-Out for Unscheduled Tasks
    # --------------------------------------------------------------------------
    def _stratus_scaler(self, unscheduled_tasks: List[pd.Series]):
        """
        Simple Stratus-style scale-out for leftover tasks.
        Acquire new instances if needed.
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

            # Sort tasks by descending resource requirement
            bin_tasks.sort(key=lambda t: max(t['CPU_request'], t['memory_request']), reverse=True)

            while bin_tasks:
                # Find the smallest group of tasks that can fit in an instance
                best_score = -1
                best_group_size = 0

                for i in range(1, len(bin_tasks) + 1):
                    candidate_group = bin_tasks[:i]
                    for _, inst_type in self.available_instance_types.iterrows():
                        if self._can_fit_group(candidate_group, inst_type):
                            score = self._calculate_score(candidate_group, inst_type)
                            if score > best_score:
                                best_score = score
                                best_group_size = i

                if best_group_size == 0:
                    # Can't fit these tasks in any instance type
                    break

                # Acquire new instance for this group
                group_to_schedule = bin_tasks[:best_group_size]
                # pick cheapest instance that can fit them
                chosen_type_idx = self._pick_cheapest_fitting_instance(group_to_schedule)
                if chosen_type_idx is None:
                    break
                chosen_type = self.available_instance_types.iloc[chosen_type_idx]
                new_inst = self._acquire_new_instance(chosen_type, bin_idx)

                # Assign tasks
                for task in group_to_schedule:
                    self._assign_task_to_instance(task, new_inst)
                bin_tasks = bin_tasks[best_group_size:]

    def _calculate_score(self, task_group: List[pd.Series], instance_type: pd.Series) -> float:
        """
        Similar to Stratus: score how well a set of tasks fits a given instance type.
        """
        total_cpu = sum(t['CPU_request'] for t in task_group)
        total_mem = sum(t['memory_request'] for t in task_group)
        cpu_frac = total_cpu / instance_type['capacity_CPU']
        mem_frac = total_mem / instance_type['capacity_memory']
        return (cpu_frac + mem_frac) / instance_type['normalized_price']

    def _can_fit_group(self, task_group: List[pd.Series], instance_type: pd.Series) -> bool:
        """
        Check if a group of tasks can fit on an instance type.
        """
        total_cpu = sum(t['CPU_request'] for t in task_group)
        total_mem = sum(t['memory_request'] for t in task_group)
        return (total_cpu <= instance_type['capacity_CPU'] and
                total_mem <= instance_type['capacity_memory'])

    def _pick_cheapest_fitting_instance(self, task_group: List[pd.Series]) -> int:
        """
        Pick the cheapest instance type that can fit all tasks in the group.
        Returns the index of the chosen instance type, or None if none fits.
        """
        total_cpu = sum(t['CPU_request'] for t in task_group)
        total_mem = sum(t['memory_request'] for t in task_group)

        best_idx = None
        cheapest_price = float('inf')
        for i, (_, inst_type) in enumerate(self.available_instance_types.iterrows()):
            if inst_type['capacity_CPU'] >= total_cpu and inst_type['capacity_memory'] >= total_mem:
                if inst_type['normalized_price'] < cheapest_price:
                    cheapest_price = inst_type['normalized_price']
                    best_idx = i
        return best_idx

    def _acquire_new_instance(self, instance_type: pd.Series, bin_idx: int) -> pd.Series:
        """
        Acquire a new instance of a given type and add it to instance_bins.
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
        self.instance_bins = pd.concat(
            [self.instance_bins, pd.DataFrame([new_instance])],
            ignore_index=True
        )
        return new_instance

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    def _calculate_bin_index(self, runtime: float) -> int:
        """
        Exponential binning of runtime (Stratus approach).
        """
        if runtime <= 0:
            return 0
        return int(np.floor(np.log2(runtime))) + 1

    def _assign_task_to_instance(self, task: pd.Series, instance: pd.Series):
        """
        Assign the given task to the specified instance and update resource usage and cost.
        """
        instance_idx = self.instance_bins.index[
            self.instance_bins['instance_ID'] == instance['instance_ID']
        ][0]

        # Update resource usage
        self.instance_bins.at[instance_idx, 'CPU_used'] += task['CPU_request']
        self.instance_bins.at[instance_idx, 'memory_used'] += task['memory_request']

        # Update cost
        old_runtime = self.instance_bins.at[instance_idx, 'runtime']
        old_timestamp = self.instance_bins.at[instance_idx, 'timestamp']
        old_finish = old_timestamp + old_runtime

        new_finish = task['timestamp'] + task['runtime']
        max_finish = max(old_finish, new_finish)

        additional_runtime = max_finish - old_finish
        self.instance_bins.at[instance_idx, 'timestamp'] = task['timestamp']
        self.instance_bins.at[instance_idx, 'runtime'] = max_finish - task['timestamp']

        if old_runtime == 0:
            # First assignment on this instance
            self.price_counter += self.instance_bins.at[instance_idx, 'price'] * task['runtime']
        else:
            if additional_runtime > 0:
                self.price_counter += self.instance_bins.at[instance_idx, 'price'] * additional_runtime

        self.tasks += 1
        # Log the task assignment
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

        # Update utilization
        self._update_utilization()

    def _update_utilization(self):
        """
        Update CPU and memory utilization based on instance_bins.
        """
        total_cpu_capacity = self.instance_bins['CPU_capacity'].sum() or 1e-9
        total_memory_capacity = self.instance_bins['memory_capacity'].sum() or 1e-9
        total_cpu_used = self.instance_bins['CPU_used'].sum()
        total_memory_used = self.instance_bins['memory_used'].sum()

        self.cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100
        self.memory_utilization = (total_memory_used / total_memory_capacity) * 100


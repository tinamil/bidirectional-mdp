from __future__ import annotations

from racetrack import Racetrack, State
from typing import List, Tuple, Union, Optional, Dict
import numpy as np

"""LAO*
1. The explicit graph G' initially consists of the start state s.

2. While the best solution graph has some nonterminal tip state:
    (a) Expand best partial solution: Expand some nonterminal tip state n of the best partial solution graph and add any new successor states to G'.
    For each new state i added to G' by expanding n, if i is a goal state then f (i) := 0; else f (i) := h(i).

    (b) Update state costs and mark best actions:
        i. Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along marked action arcs.
        (I.e., only include ancestor states from which the expanded state can be reached by following the current best solution.)

        ii. Perform dynamic programming (policy iteration or value iteration) on the states in set Z to update state costs and
        determine the best action for each state.

3. Convergence test: If policy iteration was used in step 2(b)ii, go to step 4. Else perform value iteration on the states in the best solution graph.
Continue until one of the following two conditions is met. (i) If the error bound falls below ε, go to step 4. (ii) If the best current solution graph changes so that it has an unexpanded
tip state, go to step 2.

4. Return an optimal (or ε-optimal) solution graph.
"""

#TODO: Check for wall collisions!

def search(track: Racetrack, method: str):
    return LAO(track)


p = .5


class Node:
    track = None
    nodes: Dict[bytes, Node] = dict()

    def __init__(self, parent: Optional[Node], state: State):
        self.parent = parent
        self.state = state
        self.children: List[Tuple[Node, Node]] = []
        self.heuristic = self.__heuristic_estimate()
        self.cost = 1
        self.f = self.heuristic
        self.best_child = -1

        if parent is not None and self.track.is_goal(self.parent.state, self.state):
            self.cost = 0
            self.f = 0

    @staticmethod
    def build(parent: Optional[Node], state: State) -> Node:
        if state.tobytes() in Node.nodes:
            recursive_node = Node.nodes[state.tobytes()]
            return recursive_node
        else:
            new_node = Node(parent, state)
            Node.nodes[state.tobytes()] = new_node
            return new_node

    @classmethod
    def set_track(cls, racetrack: Racetrack) -> None:
        cls.track = racetrack

    def get_actions(self) -> List[Tuple[State, State]]:
        return self.track.get_actions(self.state)

    def add_child_action(self, success: Node, failure: Node) -> None:
        self.children.append((success, failure))

    def is_terminal(self) -> bool:
        if self.parent is None:
            return (self.state[0], self.state[1]) in self.track.get_objectives()
        return self.track.is_goal(self.parent.state, self.state)

    def is_recursive_child(self, child: Node) -> bool:
        return child.parent != self

    def __heuristic_estimate(self) -> float:
        if self.is_terminal():
            return 0

        goals = self.track.get_objectives()
        best = np.infty
        for x in goals:
            delta_vector = np.array(x) - self.state[:2]

            delta_x = delta_vector[0]
            delta_y = delta_vector[1]
            velocity_x = self.state[1]
            velocity_y = self.state[2]

            if delta_x < 0:
                delta_x = -delta_x
                velocity_x = -velocity_x

            if delta_y < 0:
                delta_y = -delta_y
                velocity_y = -velocity_y

            time_x = 0
            while delta_x > 0:
                velocity_x += 1
                delta_x -= velocity_x
                time_x += 1

            time_y = 0
            while delta_y > 0:
                velocity_y += 1
                delta_y -= velocity_y
                time_y += 1

            new_time = max(time_x, time_y)
            if new_time < best:
                best = new_time

        return best

    def get_recommended_actions(self) -> Optional[Tuple[Node, Node]]:
        if self.best_child == -1:
            return None
        else:
            return self.children[self.best_child]

    @staticmethod
    def get_next_nonterminal_state(root: Node) -> Optional[Node]:
        if root.cost == 0:
            # Root is a terminal state (crossed the goal)
            return None

        best_children = root.get_recommended_actions()

        if best_children is None:
            # Root is not terminal but has not yet been expanded
            return root

        success_state, failure_state = best_children
        if not root.is_recursive_child(success_state):
            success_search = Node.get_next_nonterminal_state(success_state)
            if success_search is not None:
                # The success_state has a non-terminal child node
                root.update_f()
                return success_search

        if not root.is_recursive_child(failure_state):
            failure_search = Node.get_next_nonterminal_state(failure_state)
            if failure_search is not None:
                # Failure state has a non-terminal child node
                root.update_f()
                return failure_search

        root.update_f()
        # All recommended actions are terminal
        return None

    def update_f(self) -> None:
        new_f, best_action = self.calculate_new_f()
        self.best_child = best_action
        self.f = new_f

    def calculate_new_f(self, state_values: dict=None) -> Tuple[float, int]:
        best_action = self.best_child
        if best_action > -1:
            if state_values is None:
                best_child_action_cost = self.f
            else:
                best_child_action_cost = state_values[self]
        else:
            best_child_action_cost = np.infty
        for idx, (success, failure) in enumerate(self.children):
            if state_values is None:
                cost = self.cost + (p * success.f + (1 - p) * failure.f)
            else:
                print(success.state)
                cost = self.cost + (p * state_values[success] + (1 - p) * state_values[failure])
            if cost < best_child_action_cost:
                best_child_action_cost = cost
                best_action = idx
        return best_child_action_cost, best_action


def LAO(track: Racetrack):
    """
    An efficient version of LAO* that combines backups with solution expansion
    """
    Node.set_track(track)

    '''
    1. The explicit graph G' initially consists of the start state s.
    '''
    start_g = Node.build(None, track.get_start()[0])
    next_state = start_g

    finished = False
    while not finished:
        '''
           2. Expand best partial solution, update state costs, and mark best actions: While the best solution graph has
              some nonterminal tip state, perform a depth-first search of the best partial solution graph. For each visited
              state i, in postorder traversal:
                  (a) If state i is not expanded, expand it.
                  (b) Set f (i) := \min{a \in A(i)} [c_i(a) + \sum_j p_{ij}(a)f(j)] and mark the best action for i.  (When 
                  determining the best action resolve ties arbitrarily, but give preference to the currently marked action.)
           '''
        while next_state is not None:
            actions = next_state.get_actions()
            for next_success, next_failure in actions:
                next_state.add_child_action(Node.build(next_state, next_success), Node.build(next_state, next_failure))
            next_state.update_f()
            next_state = Node.get_next_nonterminal_state(start_g)

        '''
        3. Convergence test: Perform value iteration on the states in the best solution graph. Continue until one of the
              following two conditions is met:
                    (i) If the error bound falls below ε, go to step 4.
                    (ii) If the best solution graph changes so that it has an unexpanded tip state, go to step 2.
        '''
        result = value_iteration(start_g)
        if result is not None:
            finished = True

    '''
    4. Return an ε-optimal solution graph
    '''

    return [], 0


def initialize_mdp_states(node: Node, values: dict) -> None:
    values[node] = node.f
    if node.best_child != -1:
        success, fail = node.get_recommended_actions()
        if not node.is_recursive_child(success):
            initialize_mdp_states(success, values)
        if not node.is_recursive_child(fail):
            initialize_mdp_states(fail, values)


def update_mdp_states(node: Node, values: dict, old_values, delta) -> float:
    values[node], best_action = node.calculate_new_f(old_values)
    node.best_child = best_action
    delta = max(delta, abs(values[node] - old_values[node]))
    if node.best_child != -1:
        success, fail = node.get_recommended_actions()
        if not node.is_recursive_child(success):
            delta = update_mdp_states(success, values, old_values, delta)
        if not node.is_recursive_child(fail):
            delta = update_mdp_states(fail, values, old_values, delta)
    return delta


def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration."""
    U1 = dict()
    initialize_mdp_states(mdp, U1)

    while Node.get_next_nonterminal_state(mdp) is None:
        U = U1.copy()
        delta = 0
        delta = update_mdp_states(mdp, U1, U, delta)
        if delta < epsilon:
            return U
    return None

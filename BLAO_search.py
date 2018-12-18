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


def search(track: Racetrack, method: str):
    return BLAO(track)


p = .5


class Node:
    track = None
    forward_nodes: Dict[bytes, Node] = dict()
    backward_nodes: Dict[bytes, Node] = dict()

    def __init__(self, parent: Optional[Node], state: State, forward: bool):
        self.parent = parent
        self.state = state
        self.children: List[Tuple[Node, Node]] = []
        self.reverse_children: List[Tuple[Node, Node]] = []
        self.heuristic = self.__heuristic_estimate(forward)
        self.cost = 1
        self.f = self.heuristic
        self.best_child = -1
        self.forward = forward

        if forward:
            if parent is not None and self.track.is_goal(self.parent.state, self.state):
                self.cost = 0
                self.f = 0
        else:
            if parent is not None and self.track.is_start(self.parent.state, self.state):
                self.cost = 0
                self.f = 0

    @staticmethod
    def build(parent: Optional[Node], state: State, forward: bool) -> Node:
        if forward:
            if state.tobytes() in Node.forward_nodes:
                recursive_node = Node.forward_nodes[state.tobytes()]
                return recursive_node
            else:
                new_node = Node(parent, state, forward)
                Node.forward_nodes[state.tobytes()] = new_node
                return new_node
        else:
            if state.tobytes() in Node.backward_nodes:
                recursive_node = Node.backward_nodes[state.tobytes()]
                return recursive_node
            else:
                new_node = Node(parent, state, forward)
                Node.backward_nodes[state.tobytes()] = new_node
                return new_node

    @classmethod
    def set_track(cls, racetrack: Racetrack) -> None:
        cls.track = racetrack

    def get_actions(self, forward):
        return self.track.get_actions(self.state, forward)

    def add_child_action(self, success: Node, failure: Node) -> None:
        self.children.append((success, failure))
        success.add_reverse_children(self)
        failure.add_reverse_children(self)

    def add_reverse_children(self, predecessor) -> None:
        self.reverse_children.append(predecessor)

    def is_terminal(self, forward: bool) -> bool:
        if self.parent is None:
            if forward:
                return self.state[2] == 0 and self.state[3] == 0 and (self.state[0], self.state[1]) in self.track.get_objectives()
            else:
                for x in self.track.get_start():
                    if np.array_equal(x, self.state):
                        return True
                return False
        return self.track.is_goal(self.parent.state, self.state)

    def __heuristic_estimate(self, forward: bool) -> float:
        if self.is_terminal(forward):
            return 0

        if forward:
            goals = self.track.get_objectives()
        else:
            goals = self.track.get_start()

        best = np.infty
        for x in goals:
            delta_vector = np.array(x[:2]) - self.state[:2]

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
    def get_next_nonterminal_state(root: Node, seen_nodes: set, other_direction_nodes: set) -> Optional[Node]:
        seen_nodes.add(root)

        if root in other_direction_nodes:
            return None

        if root.cost == 0:
            # Root is a terminal state (crossed the goal)
            return None

        best_children = root.get_recommended_actions()

        if best_children is None:
            # Root is not terminal but has not yet been expanded
            return root

        success_state, failure_state = best_children
        if success_state not in seen_nodes:
            success_search = Node.get_next_nonterminal_state(success_state, seen_nodes, other_direction_nodes)
            if success_search is not None:
                # The success_state has a non-terminal child node
                root.update_f()
                return success_search

        if failure_state not in seen_nodes:
            failure_search = Node.get_next_nonterminal_state(failure_state, seen_nodes, other_direction_nodes)
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

    def calculate_new_f(self) -> Tuple[float, int]:
        # best_action = self.best_child
        # if best_action > -1:
        #     best_child_action_cost = self.f
        # else:
        best_child_action_cost = np.infty
        best_action = -1
        for idx, (success, failure) in enumerate(self.children):
            cost = self.cost + (p * success.f + (1 - p) * failure.f)
            if cost < best_child_action_cost:
                best_child_action_cost = cost
                best_action = idx
        return best_child_action_cost, best_action

    def calculate_new_f_from_old_state(self, state_values, delta, new_values) -> Tuple[float, int, float]:
        # best_action = self.best_child
        # if best_action > -1:
        #     best_child_action_cost = state_values[self]
        # else:
        best_child_action_cost = np.infty
        best_action = -1
        for idx, (success, failure) in enumerate(self.children):
            if success.f is np.infty or failure.f is np.infty:
                print("Infinity f value!", success.state, failure.state)
            if success not in state_values:
                state_values[success] = success.f
                new_values[success] = success.f
                delta = max(delta, success.f)
            if failure not in state_values:
                state_values[failure] = failure.f
                new_values[failure] = failure.f
                delta = max(delta, failure.f)
            cost = self.cost + (p * state_values[success] + (1 - p) * state_values[failure])
            if cost < best_child_action_cost:
                best_child_action_cost = cost
                best_action = idx
        return best_child_action_cost, best_action, delta


def BLAO(track: Racetrack):
    """
    An efficient version of LAO* that combines backups with solution expansion
    """
    Node.set_track(track)

    '''
    1. The explicit graph G' initially consists of the start state s and goal state g
    '''
    start_g = Node.build(None, track.get_start()[0], True)

    obj_x, obj_y = track.get_objectives()[0]
    end_state = np.zeros(4, dtype=int)
    end_state[0] = obj_x
    end_state[1] = obj_y
    end_g = Node.build(None, end_state, False)

    next_state = start_g
    finished = False
    #go_forward = True
    go_forward = False
    while not finished:
        '''
            2. Forward and backward started concurrently (alternately in this case)
            3. While the best solution has some nonterminal leaf state:
                a) Expand some nonterminal leaf state s' of the graph G' and check whether any of its successors
                have already been expanded or whether it is a terminal state.  If it has already been expanded or is
                terminal, then go to step 4.  Otherwise add the successors to the graph.
                b) During forward search, update the value of each state based on the value of its successors
                that can be reached by performing the best action.  During backward search, update the value of the
                state based on the value of the states from which this state can be reached by performing their respective
                best actions.  Mark the best actions.
        '''
        while next_state is not None:
            actions = next_state.get_actions(go_forward)
            if go_forward:
                for next_success, next_failure in actions:
                    next_state.add_child_action(Node.build(next_state, next_success, go_forward), Node.build(next_state, next_failure, go_forward))
                next_state.update_f()
            else:
                next_state.update_f()
                for backward_expansion_state in actions:
                    prev_state = Node.build(None, backward_expansion_state, go_forward)
                    if prev_state.best_child == -1:
                        prev_actions = prev_state.get_actions(True)
                        for next_success, next_failure in prev_actions:
                            prev_state.add_child_action(Node.build(None, next_success, True), Node.build(None, next_failure, True))
                    prev_state.update_f()

            #go_forward = start_g.f < end_g.f

            if go_forward:
                forward_nodes = set()
                next_state = Node.get_next_nonterminal_state(start_g, forward_nodes, set())
                other_direction_nodes = forward_nodes
            else:
                backward_nodes = set()
                next_state = Node.get_next_nonterminal_state(end_g, backward_nodes, set())
                other_direction_nodes = backward_nodes


        '''
        4. Convergence test: Perform value iteration on the states in the best solution graph. Continue until one of the
              following two conditions is met:
                    (i) If the error bound falls below ε, go to step 5.
                    (ii) If the best solution graph changes so that it has an unexpanded tip state, go to step 3.
        '''
        next_state = value_iteration(start_g, end_g, go_forward)
        if next_state is None:
            finished = True

    '''
    5. Return an ε-optimal solution graph
    '''
    return start_g, end_g


def update_mdp_states(node: Node, values: dict, old_values: dict, delta: float, seen_nodes: set) -> float:
    seen_nodes.add(node)
    new_f, best_action, delta = node.calculate_new_f_from_old_state(old_values, delta, values)
    if best_action != -1:
        values[node] = new_f
        node.best_child = best_action
    else:
        values[node] = node.f
        if node not in old_values:
            old_values[node] = node.f
    delta = max(delta, abs(values[node] - old_values[node]))
    if node.best_child != -1:
        success, fail = node.get_recommended_actions()
        if success not in seen_nodes:
            delta = update_mdp_states(success, values, old_values, delta, seen_nodes)
        if fail not in seen_nodes:
            delta = update_mdp_states(fail, values, old_values, delta, seen_nodes)
    return delta


def get_optimum_nodes(node: Node, seen_nodes: set):
    seen_nodes.add(node)
    if node.best_child != -1:
        success, fail = node.get_recommended_actions()
        if success not in seen_nodes:
            get_optimum_nodes(success, seen_nodes)
        if fail not in seen_nodes:
            get_optimum_nodes(fail, seen_nodes)


def value_iteration(forward_mdp, backward_mdp, go_forward, epsilon=0.01):
    """Solving an MDP by value iteration."""
    U1_forward = dict()
    U1_backward = dict()
    nonterminal_node = None
    while nonterminal_node is None:
        U = U1_forward.copy()
        seen_nodes = set()
        delta = 0
        delta = update_mdp_states(forward_mdp, U1_forward, U, delta, seen_nodes)

        U = U1_backward.copy()
        seen_nodes = set()
        delta = update_mdp_states(backward_mdp, U1_backward, U, delta, seen_nodes)

        if delta < epsilon:
            return None

        #best_forward_nodes = set()
        #best_backward_nodes = set()
        #get_optimum_nodes(forward_mdp, best_forward_nodes)
        #get_optimum_nodes(backward_mdp, best_backward_nodes)

        forward_nonterminal = Node.get_next_nonterminal_state(forward_mdp, set(), set())
        backward_nonterminal = Node.get_next_nonterminal_state(backward_mdp, set(), set())

        #backward_nonterminal = None
        # if go_forward and forward_nonterminal is not None:
        nonterminal_node = forward_nonterminal
        # elif go_forward and backward_nonterminal is not None:
        #     nonterminal_node = backward_nonterminal
        #     go_forward = False
        # elif not go_forward and backward_nonterminal is not None:
        #     nonterminal_node = backward_nonterminal
        # elif not go_forward and forward_nonterminal is not None:
        #     nonterminal_node = forward_nonterminal
        #     go_forward = True

    return nonterminal_node

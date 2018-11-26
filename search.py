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
    return LAO(track)


p = .5


class Node:
    track = None
    nodes: Dict[State, Node] = dict()

    def __init__(self, parent: Optional[Node], state: State):
        self.parent = parent
        self.state = state
        self.children: List[Tuple[Node, Node]] = []
        self.heuristic = self.__heuristic_estimate()
        self.cost = self.heuristic
        self.best_child = -1

    #TODO: Use this to build recursive connections for LAO? Maybe
    @classmethod
    def get_node(cls, parent: Optional[Node], state: State):
        node = None
        if state in cls.nodes:
            node = cls.nodes[state]
        else:
            node = Node(parent, state)

    @classmethod
    def set_track(cls, racetrack: Racetrack) -> None:
        cls.track = racetrack

    def get_actions(self) -> List[Tuple[State, State]]:
        return Racetrack.get_actions(self.state)

    def add_child_action(self, success: Node, failure: Node) -> None:
        self.children.append((success, failure))

    def is_terminal(self) -> bool:
        if self.parent is None:
            return (self.state[0], self.state[1]) in self.track.get_objectives()
        return self.track.is_goal(self.parent.state, self.state)

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

            # TODO: Consider replacing with a closed form solution (continuous if discrete doesn't exist)

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
        success_search = Node.get_next_nonterminal_state(success_state)
        if success_search is not None:
            # The success_state has a non-terminal child node
            return success_search

        failure_search = Node.get_next_nonterminal_state(failure_state)
        if failure_search is not None:
            # Failure state has a non-terminal child node
            return failure_search

        # All recommended actions are terminal
        return None


def LAO(track: Racetrack):

    Node.set_track(track)

    start_g = Node(None, track.get_start()[0])

    next_state = start_g
    while next_state is not None:

        # Expand the next state with its possible actions
        actions = next_state.get_actions()
        for next_success, next_failure in actions:
            next_state.add_child_action(Node(next_state, next_success), Node(next_state, next_failure))

        # Update state costs

        # 1. Add next_state and all ancestors to ancestors set
        ancestors = set()
        node = next_state
        while node is not None:
            ancestors.add(node)
            node = node.parent

        # 2. Repeat until z is empty:
        #    A. Remove from Z a state i such that no descendent of i in G' occurs in Z (Not guaranteed if looping is allowed)
        #    B. Set i.cost = min cost expected value of action (1 + p * success.heuristic + (1-p) failure.heuristic)

        next_state = Node.get_next_nonterminal_state(start_g)
    return [], 0


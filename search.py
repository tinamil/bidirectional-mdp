from __future__ import annotations
from racetrack import Racetrack, State
from typing import List, Tuple, Union, Optional
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



class Node:
    def __init__(self, parent: Optional[Node], state: State):
        self.parent = parent
        self.state = state
        self.children = []

    def get_actions(self) -> List[Node]:
        neighbors = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                new_neighbor = Node(self, self.state + [self.state[2] + x, self.state[3] + y, x, y])
                neighbors.append(new_neighbor)
        return neighbors

    def add_child(self, child: Node):
        self.children.append(child)


def LAO(track: Racetrack):
    start_g = Node(None, track.get_start()[0])
    neighbors = start_g.get_actions()
    done = False

    return [], 0

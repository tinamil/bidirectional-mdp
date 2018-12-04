"""A race track of any shape is drawn on graph paper, with a starting line at one end and a finish line at the other consisting of designated squares. Each square within the
boundary of the track is a possible location of the car. At the start of each of a sequence of trials, the car is placed on the starting line at
a random position, and moves are made in which the car attempts to move down the track toward the finish line. Acceleration and deceleration are simulated as follows. If
in the previous move the car moved h squares horizontally and u squares vertically, then the present move can be h’ squares vertically and v’ squares horizontally, where
the difference between h’ and h is - 1, 0, or 1, and the difference between v’ and v is - 1, 0, or 1. This means that the car can maintain its speed in either dimension, or it
can slow down or speed up in either dimension by one square per move. If the car hits the track boundary, we move it back to a random position on the starting line, reduce
its velocity to zero (i.e., h’ - h and U’ - c’ are considered to be zero), and continue the trial. The objective is to learn to control the car so that it crosses the finish line in as
few moves as possible.

In addition to the difficulty of discovering faster ways to reach the finish line, it is very easy for the car to gather too much speed to negotiate the track’s curves. To make
matters worse, we introduce a random factor into the problem. With a probability p, the actual accelerations or decelerations at a move are zero independently of the intended
accelerations or decelerations. Thus, 1 -p is the probability that the controller’s intended actions are executed. One might think of this as simulating driving on a track that is
unpredictably slippery so that sometimes braking and throttling up have no effect on the car’s velocity.  -Barto et al. - 1995 - Learning to act using real-time dynamic programming

Current state is a quadruple tuple of integers (x, y, x', y') representing horizontal and vertical position and horizontal and vertical velocity.  Valid actions at time t are 9 different velocity
changes in the range -1 to +1: i.e. {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), or (1, 1)}

Velocity at step t+1 is x' + (-1 to 1), position is position + new velocity with probability p
Velocity changes are ignored (track slippage) and position is position + old velocity with probability (1-p)

If the line connecting the old and new position in 2D space intersects a track boundary then a collision occurs.  The velocity is set to 0 and the car is moved
back to the starting position.

If the finish line is crossed then an absorbing goal state is reached and the MDP is terminated.

All actions are unit cost, i.e. c(state) = 1

Only policies that are guaranteed to cross the finish line are considered to avoid the need to use discounting to prevent infinite loops.  Therefore, the infinite horizon cost is the
expected number of moves for the car to cross the finish line from state i.  An optimal policy minimizes the number of expected moves to reach the finish line.

"""

import re
import copy
import numpy as np
from typing import Tuple, List

Position = Tuple[int, int]
Velocity = Tuple[int, int]
Dimension = Tuple[int, int]
State = np.ndarray


class Racetrack:
    def __init__(self, filename: str) -> None:
        '''
        Given a file will read in and initialize the racetrack.  File must have exactly one start char and at least one reachable goal character
        :param filename:
        '''
        self.__wallChar = '%'
        self.__startChar = 'P'
        self.__objectiveChar = '.'
        self.__start: List[State] = []
        self.__objective: List[Position] = []

        with open(filename) as f:
            lines = f.readlines()

        """Strip out any lines that only consist of whitespace"""
        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        """Remove all newlines"""
        lines = [list(line.strip('\n')) for line in lines]

        self.track = np.array(lines)

        self.rows = self.track.shape[0]
        self.cols = self.track.shape[1]

        for x in range(self.rows):
            for y in range(self.cols):
                if self.track[x, y] == self.__startChar:
                    self.__start.append(np.array([x, y, 0, 0]))
                elif self.track[x, y] == self.__objectiveChar:
                    self.__objective.append((x, y))

    def is_wall(self, x: int, y: int) -> bool:
        return self.track[x][y] == self.__wallChar

    def __is_objective(self, x: int, y: int) -> bool:
        return (x, y) in self.__objective

    def get_objectives(self) -> List[Position]:
        return copy.deepcopy(self.__objective)

    def get_start(self) -> List[State]:
        return self.__start

    def get_dimensions(self) -> Dimension:
        return self.rows, self.cols

    def get_actions(self, state: State) -> List[Tuple[State, State]]:
        #TODO: Start assumes index 0
        neighbors = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                new_neighbor = state + np.array([state[2] + x, state[3] + y, x, y])
                other_new_neighbor = state + np.array([state[2], state[3], 0, 0])
                if self.is_collision(state, new_neighbor):
                    new_neighbor = self.get_start()[0]
                if self.is_collision(state, other_new_neighbor):
                    other_new_neighbor = self.get_start()[0]
                neighbors.append((new_neighbor, other_new_neighbor))
        return neighbors

    def __is_collision(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        nodes = self.raytrace(x0, y0, x1, y1)
        for x, y in nodes:
            if self.is_wall(x, y):
                return True
        return False

    def __is_goal(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        nodes = self.raytrace(x0, y0, x1, y1)
        for x, y in nodes:
            if self.__is_objective(x, y):
                return True
        return False

    def is_goal(self, state1: State, state2: State):
        return self.__is_goal(state1[0], state1[1], state2[0], state2[1])

    def is_collision(self, state1: State, state2: State):
        return self.__is_collision(state1[0], state1[1], state2[0], state2[1])

    @staticmethod
    def raytrace(x0: int, y0: int, x1: int, y1: int) -> List[Position]:
        """
        Retuns a list of all positions (x,y) on a line between (x0, y0) and (x1, y1), inclusive
        http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        n = 1 + dx + dy
        if x1 > x0:
            x_inc = 1
        else:
            x_inc = -1
        if y1 > y0:
            y_inc = 1
        else:
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2
        tiles_visited = []
        for i in range(n, 0, -1):
            tiles_visited.append((x, y))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return tiles_visited

########################################################
#
# CMPSC 441: Homework 3
#
########################################################


student_name = 'John Janisheski'
student_email = 'jqj5487@psu.edu'


########################################################
# Import
########################################################

from hw3_utils import *
from collections import deque
import math
# Add your imports here if used


##########################################################
# 1. Best-First, Uniform-Cost, A-Star Search Algorithms
##########################################################


def best_first_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    frontier = deque([node])         # queue: popleft/append-sorted
    explored = [problem.init_state]  # used as "visited"
    while frontier.__len__() > 0:
        u = frontier.popleft()
        explored.append(u)
        if problem.goal_test(u.state):
            return u
        lst = u.expand(problem)
        for x in lst:
            if not explored.__contains__(x):
                frontier.append(x)
        d = dict()
        for x in frontier:
            if x.heuristic in d:
                d[x.heuristic].append(x)
            else:
                d[x.heuristic] = [x]

        frontier.clear()
        for x in sorted(d.keys()):
            for y in d[x]:
                frontier.append(y)


def uniform_cost_search(problem):
    node = Node(problem.init_state)
    frontier = deque([node])         # queue: popleft/append-sorted
    explored = []                    # used as "expanded" (not "visited")

    while frontier.__len__() > 0:
        u = frontier.popleft()
        explored.append(u)
        if problem.goal_test(u.state):
            return u
        lst = u.expand(problem)
        for x in lst:
            frontier.append(x)
        d = dict()
        for x in frontier:
            g = x.path_cost
            if g in d:
                d[g].append(x)
            else:
                d[g] = [x]

        frontier.clear()
        for x in sorted(d.keys()):
            for y in d[x]:
                frontier.append(y)


def a_star_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    frontier = deque([node])         # queue: popleft/append-sorted
    explored = []                    # used as "expanded" (not "visited")
    while frontier.__len__() > 0:
        u = frontier.popleft()
        explored.append(u)
        if problem.goal_test(u.state):
            return u
        lst = u.expand(problem)
        for x in lst:
            if not explored.__contains__(x):
                frontier.append(x)
        d = dict()
        for x in frontier:
            if x.heuristic + x.path_cost in d:
                d[x.heuristic + x.path_cost].append(x)
            else:
                d[x.heuristic + x.path_cost] = [x]

        frontier.clear()
        for x in sorted(d.keys()):
            for y in d[x]:
                frontier.append(y)


##########################################################
# 2. N-Queens Problem
##########################################################


class NQueensProblem(Problem):
    """
    The implementation of the class NQueensProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """
    
    def __init__(self, n):
        super().__init__(tuple([-1] * n))
        self.n = n
        

    def actions(self, state):
        if state[-1] != -1:   # if all columns are filled
            return []         # then no valid actions exist
        
        valid_actions = list(range(self.n))
        col = state.index(-1) # index of leftmost unfilled column
        for row in range(self.n):
            for c, r in enumerate(state[:col]):
                if self.conflict(row, col, r, c) and row in valid_actions:
                    valid_actions.remove(row)
                    
        return valid_actions

        
    def result(self, state, action):
        col = state.index(-1) # leftmost empty column
        new = list(state[:])  
        new[col] = action     # queen's location on that column
        return tuple(new)

    
    def goal_test(self, state):
        if state[-1] == -1:   # if there is an empty column
            return False     # then, state is not a goal state

        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    return False
        return True

    
    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1-row2) == abs(col1-col2)

    
    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from start state to to_state via from_state.
        The path from start_state to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        return cost + 1


    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the total number of conflicts in the given
        state as a heuristic value for the state.
        """
        h = 0
        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    h = h+1
        return h


##########################################################
# 3. Graph Problem
##########################################################
class GraphProblem(Problem):
    """
    The implementation of the class GraphProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """
    
    
    def __init__(self, init_state, goal_state, graph):
        super().__init__(init_state, goal_state)
        self.graph = graph

        
    def actions(self, state):
        """Returns the list of adjacent states from the given state."""
        return list(self.graph.get(state).keys())

    
    def result(self, state, action):
        """Returns the resulting state by taking the given action.
            (action is the adjacent state to move to from the given state)"""
        return action

    
    def goal_test(self, state):
        return state == self.goal_state

    
    def g(self, cost, from_state, action, to_state):
        """
        Returns the path cost from root to to_state.
        Note that the path cost from the root to from_state
        is the give cost and the given action taken at from_state
        will lead you to to_state with the cost associated with
        the action.
        """
        return cost + romania_roads[from_state][action]
    

    def h(self, state):
        """
        Returns the heuristic value for the given state. Heuristic
        value of the state is calculated as follows:
        1. if an attribute called 'heuristics' exists:
           - heuristics must be a dictionary of states as keys
             and corresponding heuristic values as values
           - so, return the heuristic value for the given state
        2. else if an attribute called 'locations' exists:
           - locations must be a dictionary of states as keys
             and corresponding GPS coordinates (x, y) as values
           - so, calculate and return the straight-line distance
             (or Euclidean norm) from the given state to the goal
             state
        3. else
           - cannot find nor calculate heuristic value for given state
           - so, just return a large value (i.e., infinity)
        """
        if hasattr(self.graph, 'heuristics'):
            return self.graph.heuristics[state]
        elif hasattr(self.graph, 'locations'):
            x = abs(self.graph.locations[state][0] - self.graph.locations[self.goal_state][0])
            y = abs(self.graph.locations[state][1] - self.graph.locations[self.goal_state][1])
            c = math.sqrt(pow(x, 2) + pow(y, 2))
            return c
        else:
            return math.inf


##########################################################
# 4. Eight Puzzle
##########################################################
class EightPuzzle(Problem):

    def __init__(self, init_state, goal_state=(1,2,3,4,5,6,7,8,0)):
        super().__init__(init_state, goal_state)
    

    def actions(self, state):
        c = 0
        d = {0: ['DOWN', 'RIGHT'], 1: ['DOWN', 'LEFT', 'RIGHT'],2: ['DOWN', 'LEFT'],
             3: ['UP', 'DOWN', 'RIGHT'], 4: ['UP', 'DOWN', 'LEFT', 'RIGHT'], 5: ['UP', 'DOWN', 'LEFT'],
             6: ['UP', 'RIGHT'], 7: ['UP', 'LEFT', 'RIGHT'], 8: ['UP', 'LEFT']}
        for x in state:
            if x == 0:
                break
            c = c + 1
        return d[c]
    
    def result(self, state, action):
        c = state.index(0)
        t = list(state)
        if action.__eq__('UP'):
            tmp = t[c - 3]
            t[c - 3] = 0
            t[c] = tmp
        elif action.__eq__('DOWN'):
            tmp = t[c + 3]
            t[c + 3] = 0
            t[c] = tmp
        elif action.__eq__('LEFT'):
            tmp = t[c - 1]
            t[c - 1] = 0
            t[c] = tmp
        else:
            tmp = t[c + 1]
            t[c + 1] = 0
            t[c] = tmp
        return tuple(t)

    def goal_test(self, state):
        return state.__eq__((0, 1, 2, 3, 4, 5, 6, 7, 8))

    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from root to to_state via from_state.
        The path from root to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        return cost + 1

    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the sum of the Manhattan distances of misplaced
        tiles to their final positions.
        """
        running = 0
        c = 0
        for x in state:
            if (state[c] == 0 and state[8] == 1) or (state[0] == 0 and state[c] == 1):
                running = running + 4
            elif not state[c] == c + 1:
                if c == 8:
                    c = 0
                w = state.index(c+1)
                if abs(c - w) == 3 or abs(c - w) == 1:
                    running = running + 1
                elif abs(c - w) == 2 or abs(c - w) == 4 or abs(c - w) == 6:
                    running = running + 2
                elif abs(c - w) == 5 or abs(c - w) == 7:
                    running = running + 3
                elif abs(c - w) == 8:
                    running = running + 4
            c = c + 1
        return running

##########################################################
# Test Cases
##########################################################
#best_first_search(NQueensProblem(0))
# p = NQueensProblem(8)
# print(p.h((7,1,3,0,6,-1,-1,-1)))
# print(best_first_search(p).solution())
# print(uniform_cost_search(p).solution())
# print(a_star_search(p).solution())


# romania_map = Graph(romania_roads, False)
# romania_map.locations = romania_city_positions
# romania = GraphProblem('Arad', 'Bucharest', romania_map)
# g = GraphProblem('Arad', 'Bucharest', romania_map)
# print(best_first_search(g).solution())
# print(uniform_cost_search(g).solution())
# print(a_star_search(g).solution())
# print(romania.g(140, 'Sibiu', 'Rimnicu', 'Rimnicu'))
# print(romania.h('Arad'))

# puzzle = EightPuzzle((1, 0, 6, 8, 7, 5, 4, 2, 3))
# print(puzzle.init_state)
# print(puzzle.actions((6, 3, 5, 1, 8, 4, 2, 0, 7)))
# print(puzzle.result((0, 1, 2, 3, 4, 5, 6, 7, 8), 'DOWN'))
# print(puzzle.result((6, 3, 5, 1, 8, 4, 2, 0, 7), 'LEFT'))
# print(puzzle.result((3, 4, 1, 7, 6, 0, 2, 8, 5), 'UP'))
# print(puzzle.result((1, 8, 4, 7, 2, 6, 3, 0, 5), 'RIGHT'))
# print(puzzle.goal_test((6, 3, 5, 1, 8, 4, 2, 0, 7)))
# print(puzzle.goal_test((0, 1, 2, 3, 4, 5, 6, 7, 8)))
# print(puzzle.h((0, 2, 3, 4, 5, 6, 7, 8, 1)))
# print(puzzle.h((1, 2, 0, 4, 5, 3, 7, 8, 6)))
#print(puzzle.h((4, 1, 2, 6, 8, 0, 3, 5, 7)))
# e = EightPuzzle((3, 4, 1, 7, 6, 0, 2, 8, 5))
# print(best_first_search(e).solution())
# print(uniform_cost_search(e).solution())
# print(a_star_search(e).solution())


# roads = dict(S=dict(A=1, B=2), A=dict(C=1), B=dict(C=2), C=dict(G=100))
# roads_h = dict(S=90, A=100, B=88, C=100, G=0)
# roads_map = Graph(roads, True)
# roads_map.heuristics = roads_h
# problem = GraphProblem('S', 'G', roads_map)
# print(problem.h('B'))

# -------------------------------------------------
# q = NQueensProblem(8)
# res = best_first_search(q).solution()           1 point
# res = uniform_cost_search(q).solution()         1 point
# res = a_star_search(q).solution()               1 point
# print(res)
# -------------------------------------------------
# romania_map = Graph(romania_roads, False)       1 point
# g = GraphProblem('Arad', 'Bucharest', romania_map)
# res = best_first_search(g).solution()
# print(res)
# -------------------------------------------------
# romania_map = Graph(romania_roads, False)       1 point
# romania_map.locations = romania_city_positions
# g = GraphProblem('Arad', 'Bucharest', romania_map)
# res = best_first_search(g).solution()
# print(res)
# -------------------------------------------------
# Don't have the indiana Map
# g = GraphProblem('Bloomington', 'Washington_DC', indiana_map)
# res = best_first_search(g).solution()
# print(res)
# -------------------------------------------------
# romania_map = Graph(romania_roads, True)
# g = GraphProblem('Arad', 'Bucharest', romania_map)
# res = uniform_cost_search(g).solution()
# print(res)
# -------------------------------------------------


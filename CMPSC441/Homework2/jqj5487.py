########################################################
#
# CMPSC 441: Homework 2
#
########################################################


# student_name = 'John Janisheski'
# student_email = 'jqj5487@psu.edu'




########################################################
# Import
########################################################
from hw2_utils import *
from collections import deque
# Add your imports here if used


##########################################################
# 1. Uninformed Any-Path Search Algorithms
##########################################################

def depth_first_search(problem):
    
    node = Node(problem.init_state)
    frontier = deque([node])         # stack: append/pop
    explored = [problem.init_state]  # used as "visited"
    while len(frontier) > 0:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        for x in node.expand(problem):
            if frontier.count(x) == 0 and explored.count(x) == 0:
                frontier.appendleft(x)
                explored.append(x)
    return Node(None)


def breadth_first_search(problem):
    
    node = Node(problem.init_state)
    frontier = deque([node])         # queue: append/popleft
    explored = [problem.init_state]  # used as "visited"
    while len(frontier) > 0:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        for x in node.expand(problem):
            if frontier.count(x) == 0 and explored.count(x) == 0:
                frontier.append(x)
                explored.append(x)
    return Node(None)

##########################################################
# 2. N-Queens Problem
##########################################################
# Doesn't work for a test case of 2 (Should be None)


class NQueensProblem(Problem):

    def __init__(self, n):
        """
        __init__ should first initialize the parent portion of the instance by
        calling the parent’s __init__ method with the proper initial state (i.e.,n-tuple of−1’s).
        Then, it should initialize its own instance variables as needed.
        """
        t = (-1,)*n
        Problem.__init__(self, t)
        self.t = t
        self.n = n

    def actions(self, state):
        """
        actions should first locate the leftmost column to be filled, find all the valid rows on
        that column that do not attack any queens already on the board, and return the list of the valid row
        numbers on that column.
        """
        lst = list(range(self.n))
        used = list()
        c = 0
        for x in state:
            if x != -1:
                c = c + 1
            else:
                break
        for x in state:
            if c == 0:
                break
            elif x != -1:
                if used.count(x) == 0 and lst.count(x) > 0:
                    lst.remove(x)
                    used.append(x)
                if x-c >= 0 and used.count(x-c) == 0 and lst.count(x-c) > 0:
                    used.append(x-c)
                    lst.remove(x-c)
                if x+c <= self.n - 1 and used.count(x+c) == 0 and lst.count(x+c) > 0:
                    lst.remove(x+c)
                c = c-1

        return lst

    def result(self, state, action):
        """
        result returns a new state that results from executing the given action on
        the leftmost empty column in the given state.
        """
        lst = list(state)
        c = 0
        for x in state:
            if x == -1:
                lst[c] = action
                break
            c = c + 1
        return tuple(lst)

    def goal_test(self, state):
        """
        goal_test returns True if all Nqueens are placed on all columns (i.e., one queen on each column)
        such that no queen attacks another queen. Returns False otherwise.
        """
        c1 = 0
        if state.count(-1) > 0:
            return False
        for x in state:
            if x == -1:
                return False
            if state.count(x) > 1:
                return False
            c2 = 0
            for y in state[0:c1]:
                c3 = c1 - c2
                if x == (y - c3) or x == (y + c3):
                    return False
                c2 = c2 + 1
            c1 = c1 + 1
        if c1 == self.n:
            return True

        return False

##########################################################
# 3. Farmer's Problem
##########################################################
# You can assume that they won't have the state start as something like the chicken and the grain on one
# side with the farmer on the other

class FarmerProblem(Problem):
    
    def __init__(self, init_state, goal_state):
        """
        should first initialize the parent portion of the instance by calling the parent’s init method with proper
        arguments. Then, it should initialize its own instance variables as needed.
        """
        Problem.__init__(self, init_state)
        self.goal_state = goal_state

    def actions(self, state):
        """
        returns the list of valid actions that can be executed from the given state.
        """
        # state[0] = farmer
        # state[1] = grain
        # state[2] = chicken
        # state[3] = fox
        lst = ["F", "FC", "FG", "FX"]
        if state[2] == state[3] and state[2] != state[1]:
            lst.remove("F")
            lst.remove("FC")
            lst.remove("FG")
            return lst
        if state[2] == state[1]:
            lst.remove("F")
            if state[3] == state[2]:
                lst.remove("FX")
                lst.remove("FG")
            else:
                lst.remove("FX")
        if state[0] != state[2] and lst.count("FC") > 0:
            lst.remove("FC")
        if state[0] != state[3] and lst.count("FX") > 0:
            lst.remove("FX")
        if state[0] != state[1] and lst.count("FG") > 0:
            lst.remove("FG")
        return lst
    
    def result(self, state, action):
        """
        returns a new state that results from executing the given action in the given state.
        """
        # state[0] = farmer
        # state[1] = grain
        # state[2] = chicken
        # state[3] = fox
        lst = list(state)
        if action == "F":
            lst[0] = not state[0]
        if action == "FG":
            lst[1] = not state[1]
        if action == "FC":
            lst[2] = not state[2]
        else:
            lst[3] = not state[3]
        return tuple(lst)

    def goal_test(self, state):
        """
        returns True if the given state is the goal state. Returns False otherwise.
        """
        if state == self.goal_state:
            return True
        return False


##########################################################
# 4. Graph Problem
##########################################################

class GraphProblem(Problem):
    
    def __init__(self, init_state, goal_state, graph):
        """
        should first initialize the parent portion of the instance by calling the parent’s init method with proper
        arguments. Then, it should initialize its own instance variables as needed.
        """
        Problem.__init__(self, init_state)
        self.goal_state = goal_state
        self.graph = graph

    def actions(self, state):
        """
        returns the list of adjacent cities from the given city (i.e.,state)
        """
        lst = list()
        for x in self.graph.get(state):
            lst.append(x)
        return lst

    def result(self, state, action):
        """
        returns a new state that results from executing the given action in the given state. Note that state is a city
        we are currently in and action is one of the adjacent city to move to from the currently city.
        """
        return action

    def goal_test(self, state):
        """
        returns True if the given state is the goal state. Returns False otherwise.
        """
        if state == self.goal_state:
            return True
        return False


# NQueens Problem Functions
# q = NQueensProblem(4)
# b = breadth_first_search(q)
# d = depth_first_search(q)
# print(b.solution())
# print(d.solution())

# Farmer Problem Functions
# Currently each function works independently but Search functions have either infinite or Extremely long running time
# farmer = FarmerProblem((True, True, True, True), (False, True, False, False))
# b = breadth_first_search(farmer)
# d = depth_first_search(farmer)
# print(b.solution())
# print(d.solution())

# Graph Problem Functions
# Currently each function works independently but Search functions have either infinite or Extremely long running time
# romania_map = Graph(romania_roads, False)
# planner = GraphProblem("Arad", "Bucharest", romania_map)
# print(planner.graph)
# print(planner.init_state)
# print(planner.goal_state)
# print(planner.actions("Arad"))
# print(planner.actions("Sibiu"))
# print(planner.result("Arad", "Zerind"))
# print(planner.goal_test("Bucharest"))
# d = depth_first_search(planner)
# b = breadth_first_search(planner)
# print(d.solution())
# print(b.solution())

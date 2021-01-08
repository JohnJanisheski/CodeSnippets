########################################################
#
# CMPSC 441: Homework 4
#
########################################################


student_name = 'John Janisheski'
student_email = 'jqj5487@psu.edu'



########################################################
# Import
########################################################

from hw4_utils import *
import math
import random



# Add your imports here if used
from math import comb
from math import sin
import copy






################################################################
# 1. Genetic Algorithm
################################################################


def genetic_algorithm(problem, f_thres, ngen=1000):
    """
    Returns a tuple (i, sol) 
    where
      - i  : number of generations computed
      - sol: best chromosome found
    """
    population = problem.init_population()
    best = problem.fittest(population, f_thres)
    if best is not None:
        return -1, best
    for j in range(ngen):
        population = problem.next_generation(population)
        best = problem.fittest(population, f_thres)
        if best is not None:
            return j, best
    return ngen, best

  

################################################################
# 2. NQueens Problem
################################################################


class NQueensProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)
    
    def init_population(self):
        random.seed()
        r = list()
        for j in range(self.n):
            t = tuple()
            for k in range(self.g_len):
                t = t + (random.choice(self.g_bases),)
            r.append(t)
        return r
 
    def next_generation(self, population):
        tmp = list()
        tmp2 = list()
        j = 1
        for chrom in population:
            for i in range(self.g_len - j):
                tmp.append(self.crossover(chrom, population[i + j]))
            j += 1
        for new_chrom in tmp:
            tmp2.append(self.mutate(new_chrom))
        return self.select(self.n, tmp2)


    def mutate(self, chrom):
        random.seed()
        v = random.uniform(0, 1)
        t = tuple()
        if v > self.m_prob:
            return chrom
        i = random.randrange(0, self.g_len, 1)
        j = 0
        for x in chrom:
            if j != i:
                t = t + (x,)
            else:
                t = t + (random.choice(self.g_bases),)
            j += 1
        return t
    
    def crossover(self, chrom1, chrom2):
        random.seed()
        chrom3 = tuple()
        r = random.randrange(0, self.g_len, 1)
        j = 0
        for x, y in zip(chrom1, chrom2):
            if j >= r:
                chrom3 = chrom3 + (y,)
            else:
                chrom3 = chrom3 + (x,)
            j += 1
        return chrom3

    def fitness_fn(self, chrom):
        if chrom is None:
            return
        max_conflicts = comb(self.g_len, 2)
        conflicts = 0
        j = 1
        for x in chrom:
            for i in range(self.g_len - j):
                if chrom[i+j] == x + (i + 1) or chrom[i+j] == x - (i + 1) or chrom[i+j] == x:
                    conflicts += 1
            j += 1
        return max_conflicts - conflicts

    def select(self, m, population):
        a = tuple()
        b = tuple()
        c = tuple()
        r = list()
        sum1 = 0
        sum2 = 0
        iter = 0
        for x in population:
            a = a + (self.fitness_fn(x), x)
            sum1 += self.fitness_fn(x)
        for x in a:
            if iter%2 == 0:
                b = b + ((x/sum1),)
            else:
                b = b + (x,)
            iter += 1
        iter = 0
        for x in b:
            if iter % 2 == 0:
                c = c + ((x+sum2),)
                sum2 += x
            else:
                c = c + (x,)
            iter += 1
        random.seed()
        for x in range(m):
            v = random.uniform(0, 1)
            iter = 0
            for y in c:
                if iter%2 == 0:
                    if v <= y:
                        r.append(c[iter+1])
                        break
                iter += 1
        return r
    
    def fittest(self, population, f_thres=None):
        max = (0,)
        if f_thres == None:
            for x in population:
                if self.fitness_fn(x) > max[0]:
                    max = (self.fitness_fn(x), x)
            return max[1]
        else:
            for x in population:
                if self.fitness_fn(x) > max[0]:
                    max = (self.fitness_fn(x), x)
        if max[0] >= f_thres:
            return max[1]
        return None


################################################################
# 3. Function Optimaization f(x,y) = x sin(4x) + 1.1 y sin(2y)
################################################################


class FunctionProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)

    def init_population(self):
        r = list()
        random.seed()
        for i in range(self.n):
            x = random.uniform(0, self.g_bases[0])
            y = random.uniform(0, self.g_bases[1])
            t = x, y
            r.append(t)
        return r

    def next_generation(self, population):
        d = dict()
        tmp1 = list()
        tmp2 = list()
        tmp3 = list()
        for x in population:
            d[self.fitness_fn(x)] = x
        iter = 0
        for x in population:
            if iter < self.n/2:
                tmp1.append(x)
            iter += 1
        for chrom in tmp1:
            for i in range(len(tmp1) - 1):
                tmp2.append(self.crossover(chrom, population[i+1]))
        for x in tmp2:
            tmp3.append(self.mutate(x))
        for x in self.select(int(self.n - self.n/2), tmp3):
            tmp1.append(x)
        return tmp1
        
    def mutate(self, chrom):
        random.seed()
        x = random.uniform(0, 1)
        if x > self.m_prob:
            return chrom
        y = random.randrange(0, len(chrom), 1)
        if y == 0:
            return random.uniform(0, self.g_bases[y]), chrom[1]
        return chrom[0], random.uniform(0, self.g_bases[y])
        
    def crossover(self, chrom1, chrom2):
        random.seed()
        x_or_y = random.randrange(0, 2, 1)
        alpha = random.uniform(0,1)
        if x_or_y == 0:
            newx = ((1 - alpha) * chrom1[0]) + (alpha * chrom2[0])
            return newx, chrom1[1]
        newy = ((1 - alpha) * chrom1[1]) + (alpha * chrom2[1])
        return chrom1[0], newy
    
    def fitness_fn(self, chrom):
        if chrom is None:
            return
        x = chrom[0]
        y = chrom[1]
        return x * sin(4 * x) + 1.1 * y * sin(2 * y)
    
    def select(self, m, population):
        d = dict()
        d1 = dict()
        d2 = dict()
        for x in population:
            d[self.fitness_fn(x)] = x
        iter = 0
        for x, y in sorted(d.items()):
            d1[(self.n - iter)/sum(range(self.n+1))] = y
            iter += 1
        sum1 = 0.0
        for x, y in d1.items():
            sum1 = sum1 + x
            d2[sum1] = y
        r = list()
        for a in range(m):
            random.seed()
            k = random.uniform(0, 1)
            for x, y in d2.items():
                if k <= x:
                    r.append(y)
                    break
        return r

    def fittest(self, population, f_thres=None):
        best = (99999999,)
        if f_thres is None:
            for x in population:
                tmp = self.fitness_fn(x)
                if tmp < best[0]:
                    best = (tmp, x)
            return best[1]
        for x in population:
            tmp = self.fitness_fn(x)
            if tmp < best[0]:
                best = (tmp, x)
        if best[0] < f_thres:
            return best[1]
        return None



################################################################
# 4. Traveling Salesman Problem
################################################################


class HamiltonProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob, graph=None):
        super().__init__(n, g_bases, g_len, m_prob)

    def init_population(self):
        r = list()
        for x in range(self.n):
            random.shuffle(self.g_bases)
            r.append(copy.deepcopy(self.g_bases))
        return r
          
    def next_generation(self, population):
        pass
          
    def mutate(self, chrom):
        random.seed()
        x = random.uniform(0, 1)
        if x > self.m_prob:
            return chrom
        x = 0
        y = 0
        t = tuple()
        while x == y:
            x = random.randrange(0, len(chrom), 1)
            y = random.randrange(0, len(chrom), 1)
        for a in chrom:
            if chrom[x] == a:
                t = t + (chrom[y],)
            elif chrom[y] == a:
                t = t + (chrom[x],)
            else:
                t = t + (a,)
        return list(t)

    
    def crossover(self, chrom1, chrom2):
        pass

    def fitness_fn(self, chrom):
        pass

    def select(self, m, population):
        pass
        
    def fittest(self, population, f_thres=None):
        pass


########################
#  Tests for genetic algorithm

# p = NQueensProblem(10, (0, 1, 2, 3), 4, 0.2)
# i, sol = genetic_algorithm(p, 6, 1000)
# print(i, sol)
# # (2, (1, 3, 0, 2))
# print(p.fitness_fn(sol))
# # 6
# #
# p = NQueensProblem(100, (0, 1, 2, 3, 4, 5, 6, 7), 8, 0.2)
# i, sol = genetic_algorithm(p, 25, 1000)
# print(i, sol)
# # (1, (4, 1, 1, 6, 2, 3, 7, 3))
# print(p.fitness_fn(sol))
# # 25

# p = NQueensProblem(100, (0, 1, 2, 3, 4, 5, 6, 7), 8, 0.2)
# i, sol = genetic_algorithm(p, 28, 1000)
# print(i, sol)
# # (218, (2, 5, 7, 1, 3, 0, 6, 4))
# print(p.fitness_fn(sol))
# # 28

# p = FunctionProblem(12, (10, 10), 2, 0.2)
# i, sol = genetic_algorithm(p, -18, 1000)
# print(i, sol)
# # (25, (9.004375331562994, 8.533970262737833))
# print(p.fitness_fn(sol))
# # -18.128675496887553
#
# p = FunctionProblem(20, (10, 10), 2, 0.2)
# i, sol = genetic_algorithm(p, -18.5519, 1000)
# print(i, sol)
# # (386, (9.039685760225542, 8.670632807448825))
# print(p.fitness_fn(sol))
# # -18.554571709857512

# p = HamiltonProblem(100, univ_bases, 20, 0.1, univ_map)
# i, sol = genetic_algorithm(p, 15000, 1000)
# print(i, sol)
# # (19, ("Yale", "Pittsburgh", "Michigan", "Louisville", "Oklahoma", "Louisiana", "NotreDame","Wisconsin", "Oregon",
# # "BrighamYoung", "ArizonaState", "Stanford", "NewMexico","Colorado", "NorthDakota", "TexasAM", "FloridaState", "Duke'
# # , "Ohio", "Brown"))
# print(p.fitness_fn(sol))
# # 14318

# p = HamiltonProblem(200, univ_bases, 20, 0.1, univ_map)
# i, sol = genetic_algorithm(p, 12000, 1000)
# print(i, sol)
# # (48, ("FloridaState", "Duke", "Wisconsin", "NotreDame", "Louisville", "Michigan","Pittsburgh", "Brown", "Yale",
# # Ohio", "Oklahoma", "TexasAM", "Colorado", "NewMexico","ArizonaState", "Stanford", "Oregon", "BrighamYoung",
# # "NorthDakota", "Louisiana"))
# print(p.fitness_fn(sol))
# # 11966
########################

########################
#  Tests for N-Queens init
# p = NQueensProblem(10, (0, 1, 2, 3, 4, 5, 6, 7), 8, 0.1)
# print(p.n, p.g_len, p.m_prob)
# # (10, 8, 0.1)
# print(p.g_bases)
# # (0, 1, 2, 3, 4, 5, 6, 7)
#
# p = NQueensProblem(10, (0, 1, 2, 3), 4, 0.2)
# print(p.n, p.g_len, p.m_prob)
# # (10, 4, 0.2)
# print(p.g_bases)
# # (0, 1, 2, 3)
########################

########################
#  Tests for N-Queens init_population
# p = NQueensProblem(5, range(4), 4, 0.2)
# print(p.init_population())
# # [(2, 1, 2, 0), (0, 2, 2, 1), (0, 3, 1, 3), (2, 2, 1, 0), (2, 0, 1, 3)]
#
# p = NQueensProblem(5, range(8), 8, 0.2)
# print(p.init_population())
# # [(0, 7, 3, 5, 3, 6, 1, 2), (0, 7, 3, 2, 2, 6, 0, 1), (4, 5, 5, 6, 2, 7, 2, 3),(5, 1, 2, 1, 6, 2, 1, 7),
# # (3, 0, 0, 6, 0, 5, 3, 7)]
########################

########################
#  Tests for N-Queens next_generation
# p = NQueensProblem(5, range(8), 8, 0.2)
# population = [(2, 4, 7, 7, 5, 2, 4, 3), (3, 6, 6, 2, 3, 0, 0, 0), (0, 2, 6, 0, 4, 1, 2, 7), (2, 6, 2, 0, 6, 0, 5, 5), (0, 5, 0, 0, 2, 0, 1, 5)]
# print(p.next_generation(population))
# # [(2, 4, 7, 7, 6, 0, 5, 5), (3, 6, 6, 1, 3, 0, 0, 5), (3, 6, 6, 2, 3, 0, 5, 5), (2, 4, 7, 7, 5, 2, 4, 5),
# # (2, 2, 6, 0, 4, 1, 2, 7)]
########################

########################
#  Tests for N-Queens mutate
# p = NQueensProblem(10, range(8), 8, 0.8)
# print(p.mutate((4, 4, 4, 2, 6, 6, 4, 3)))
# # (4, 4, 4, 2, 1, 6, 4, 3)
#
# print(p.mutate((4, 4, 4, 2, 6, 6, 4, 3)))
# # (4, 4, 4, 2, 6, 6, 4, 3)
########################

########################
#  Tests for N-Queens crossover
# p = NQueensProblem(10, range(8), 8, 0.2)
# print(p.crossover((1, 4, 5, 6, 7, 1, 0, 2), (1, 3, 5, 1, 4, 7, 4, 4)))
# # (1, 4, 5, 6, 4, 7, 4, 4)
#
# p = NQueensProblem(10, range(15), 15, 0.2)
# print(p.crossover((7, 13, 3, 6, 10, 8, 3, 13, 0, 12, 12, 2, 14, 4, 12), (1, 1, 8, 13, 3, 8, 5, 7, 0, 11, 12, 11, 11, 1, 9)))
# # (7, 13, 3, 6, 10, 8, 5, 7, 0, 11, 12, 11, 11, 1, 9)
########################

########################
#  Tests for N-Queens fitness_fn
# p = NQueensProblem(10, range(4), 4, 0.2)
# print(p.fitness_fn((2, 3, 0, 1)))
# # 2
# print(p.fitness_fn((2, 0, 3, 1)))
# # 6
#
# p = NQueensProblem(10, range(8), 8, 0.2)
# print(p.fitness_fn((4, 6, 6, 2, 7, 6, 6, 4)))
# # 17
# print(p.fitness_fn((5, 3, 0, 4, 7, 1, 6, 2)))
# # 28
########################

########################
#  Tests for N-Queens select
# p = NQueensProblem(5, range(8), 8, 0.2)
# population = [(4, 2, 6, 4, 7, 4, 3, 4), (3, 5, 5, 1, 5, 0, 7, 7),(0, 4, 0, 3, 4, 5, 6, 6), (5, 7, 3, 1, 7, 4, 5, 7),(6, 7, 5, 7, 4, 7, 5, 7)]
# print(list(map(p.fitness_fn, population)))
# # [17, 22, 15, 21, 18]
# print(p.select(0, population))
# # []
# print(p.select(2, population))
# # [(6, 7, 5, 7, 4, 7, 5, 7), (3, 5, 5, 1, 5, 0, 7, 7)]
# print(p.select(5, population))
# # [(0, 4, 0, 3, 4, 5, 6, 6), (6, 7, 5, 7, 4, 7, 5, 7), (5, 7, 3, 1, 7, 4, 5, 7),(3, 5, 5, 1, 5, 0, 7, 7), (5, 7, 3, 1, 7, 4, 5, 7)]
# print(p.select(10, population))
# # [(0, 4, 0, 3, 4, 5, 6, 6), (3, 5, 5, 1, 5, 0, 7, 7), (5, 7, 3, 1, 7, 4, 5, 7),(4, 2, 6, 4, 7, 4, 3, 4),
# # (4, 2, 6, 4, 7, 4, 3, 4), (6, 7, 5, 7, 4, 7, 5,7),(5, 7, 3, 1, 7, 4, 5, 7), (3, 5, 5, 1, 5, 0, 7, 7),
# # (5, 7, 3, 1, 7, 4, 5,7),(6, 7, 5, 7, 4, 7, 5, 7)]
########################

########################
#  Tests for N-Queens fittest
# p = NQueensProblem(5, range(8), 8, 0.2)
# population = [(5, 4, 0, 2, 1, 1, 4, 3), (1, 4, 5, 2, 0, 1, 5, 7),(0, 2, 7, 4, 6, 0, 4, 5), (6, 3, 5, 5, 2, 3, 1, 0),(6, 5, 1, 7, 7, 2, 2, 3)]
# print(list(map(p.fitness_fn, population)))
# # [17, 22, 23, 19, 22]
# print(p.fittest(population))
# # (0, 2, 7, 4, 6, 0, 4, 5)
# print(p.fittest(population, 23))
# # (0, 2, 7, 4, 6, 0, 4, 5)
# print(p.fittest(population, 25))
########################

########################
#  Tests for Function Optimization __init__
# p = FunctionProblem(10, (5,5), 2, 0.1)
# print(p.n, p.g_len, p.m_prob)
# # (10, 2, 0.1)
# print(p.g_bases)
# # (5, 5)
#
# p = FunctionProblem(20, (10,20), 2, 0.2)
# print(p.n, p.g_len, p.m_prob)
# # (20, 2, 0.2)
# print(p.g_bases)
# # (10, 20)
########################

########################
#  Tests for function Optimization init_population
# p = FunctionProblem(3, (5,5), 2, 0.2)
# print(p.init_population())
# # [(1.069291442240965, 3.4258608103087766), (0.5814586733970367, 0.1062245125561545),
# # (3.2155287126033154, 0.26426882116695194)]
#
# p = FunctionProblem(5, (10,20), 2, 0.1)
# print(p.init_population())
# [(8.181250719670372, 14.869963460562444), (2.903310334690822, 18.61952673728084),(3.498760496405838,
# 16.46420547601544), (1.2659877772201877, 14.089176499032348),(9.823642185170762, 8.441220137200729)]
########################

########################
# #  Tests for function optimization next_generation
# p = FunctionProblem(6, (5,5), 2, 0.2)
# population =[(2.066938780637087, 1.650998608284147), (0.7928608069345605, 1.678831697303177),(3.685189771001436, 1.4280879354988107), (3.860362372295962, 1.0789325520768145),(4.673003350118171, 4.780722998076655), (2.5701433643372247, 4.9078727479819335)]
# print(list(map(p.fitness_fn, population)))
# # [1.6024472559453082, -0.41958730465195776, 3.4763217693695787,2.0048163673123365, -1.4496290991082836, -3.9980346848519694]
# print(p.next_generation(population))
# # [(4.673003350118171, 4.780722998076655), (4.673003350118171, 4.780722998076655),(2.5701433643372247, 4.9078727479819335), (2.5701433643372247, 4.9078727479819335),(4.673003350118171, 4.780722998076655), (0.7928608069345605, 1.678831697303177)]
# ########################

########################
#  Tests for function optimization mutate
# p = FunctionProblem(10, (10,20), 2, 0.5)
# c = (4.525394646650255, 13.005368584538973)
# print(p.mutate(c))
# # (4.525394646650255, 9.505694180053396)
# print(p.mutate(c))
# # (4.525394646650255, 13.005368584538973)
# print(p.mutate(c))
# # (0.3146989344654505, 13.005368584538973)
########################

########################
#  Tests for function optimization crossover
# p = FunctionProblem(10, (10, 10), 2, 0.2)
# c1 = (7.377910209443304, 3.6708167924621793)
# c2 = (2.5195159164374434, 7.248941413091508)
# print(p.crossover(c1, c2))
# # (7.377910209443304, 4.912246497034061)
# print(p.crossover(c1, c2))
# # (4.562265309739303, 3.6708167924621793)
########################

########################
#  Tests for function optimization fitness_fn
# p = FunctionProblem(10, (10, 20), 2, 0.2)
# print(p.fitness_fn((1, 1)))
# # 0.24342467420032166
# print(p.fitness_fn((4.562265309739303, 3.6708167924621793)))
# # 0.9415043737665614
# print(p.fitness_fn((2.5409025501787963, 16.828483737264428)))
# # 12.79566330899491
# print(p.fitness_fn((9.0449, 8.6643)))
# # -18.55190308788758
########################

########################
#  Tests for function optimization select
# p = FunctionProblem(6, (5,5), 2, 0.2)
# population = [(3.2785989393078507, 1.5854499726282851), (2.797299697316915, 1.6070456140483396), (1.7691382088986807, 2.893332226588736), (4.718397971920581, 0.5107148205878392), (1.6729300262695035, 1.1621142165573013), (4.83467584947063, 4.824603641199292)]
# print(list(map(p.fitness_fn, population)))
# # [1.657054107957212, -2.87307443334604, -0.2552252197910254, 0.5925228026882771,1.5969373196316696, 1.090598835733723]
# print(p.select(0, population))
# # []
# print(p.select(2, population))
# # [(4.83467584947063, 4.824603641199292), (1.7691382088986807, 2.893332226588736)]
# print(p.select(6, population))
# # [(1.7691382088986807, 2.893332226588736), (3.2785989393078507, 1.5854499726282851),(2.797299697316915,
# # 1.6070456140483396), (4.718397971920581, 0.5107148205878392),(1.7691382088986807, 2.893332226588736),
# # (1.7691382088986807, 2.893332226588736)]
# print(p.select(10, population))
# # [(1.7691382088986807, 2.893332226588736), (2.797299697316915, 1.6070456140483396),(2.797299697316915,
# # 1.6070456140483396), (1.6729300262695035, 1.1621142165573013),(1.7691382088986807, 2.893332226588736),
# # (1.7691382088986807, 2.893332226588736),(1.7691382088986807, 2.893332226588736), (2.797299697316915,
# # 1.6070456140483396),(4.83467584947063, 4.824603641199292), (1.6729300262695035, 1.1621142165573013)]
########################

########################
#  Tests for function optimization fittest
# p = FunctionProblem(6, (5,5), 2, 0.2)
# population = [(2.066938780637087, 1.650998608284147),(3.860362372295962, 1.0789325520768145),(2.5701433643372247, 4.9078727479819335),(4.673003350118171, 4.780722998076655),(3.685189771001436, 1.4280879354988107),(0.7928608069345605, 1.678831697303177)]
# print(list(map(p.fitness_fn, population)))
# # [1.6024472559453082, 2.0048163673123365, -3.9980346848519694,-1.4496290991082836, 3.4763217693695787, -0.41958730465195776]
# print(p.fittest(population))
# # (2.5701433643372247, 4.9078727479819335)
# print(p.fittest(population, -3.9))
# # (2.5701433643372247, 4.9078727479819335)
# print(p.fittest(population, -5.0))
#
########################

########################
#  Tests Hamilton init
# p = HamiltonProblem(5, test_bases, 5, 0.2, test_map)
# print(p.n, p.g_len, p.m_prob)
# # (5, 5, 0.2)
# print(p.g_bases)
# # ["A", "B", "C", "D", "E"]
########################

########################
#  Tests for Hamilton init_population
# p = HamiltonProblem(5, test_bases, 5, 0.2, test_map)
# print(p.init_population())
# # [("D", "E", "C", "B", "A"), ("A", "E", "C", "B", "D"), ("A", "C", "B", "E", "D"),("C", "D", "A", "E", "B"), ("D", "C", "A", "B", "E")]
########################

########################
#  Tests for Hamilton next_generation
########################

########################
#  Tests for Hamilton mutate
# p = HamiltonProblem(5, test_bases, 5, 0.5, test_map)
# print(p.mutate(("D", "E", "C", "B", "A")))
# # ("D", "E", "C", "B", "A")
# print(p.mutate(("D", "E", "C", "B", "A")))
# # ("D", "C", "E", "B", "A")
########################

########################
#  Tests for Hamilton crossover
# p = HamiltonProblem(10, ["C1", "C2", "C3", "C4", "C5", "C6"], 6, 0.2)
# print(p.crossover(("C4","C1","C5","C3","C2","C6"), ("C3","C4","C6","C2","C1","C5")))
# # ("C3", "C4", "C5", "C2", "C1", "C6")
#
# p = HamiltonProblem(5, test_bases, 5, 0.5, test_map)
# print(p.crossover(("D", "E", "C", "B", "A"), ("A", "E", "C", "B", "D")))
# # ("A", "E", "C", "B", "D")
# print(p.crossover(("A", "E", "C", "B", "D"), ("A", "C", "B", "E", "D")))
# # ("A", "C", "B", "E", "D")
# # #("C3", "C4", "C5", "C2", "C1", "C6")
########################

########################
#  Tests for Hamilton next_generation
########################
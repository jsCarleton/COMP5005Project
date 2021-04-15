from scipy.stats import norm
from scipy.stats import expon
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# functions to create probability distributions

def get_exp_dist(n_floors):
    x = np.arange(1, n_floors+1)
    prob = ss.expon.cdf(x + 0.5, loc=0, scale=n_floors/4)\
                     - ss.expon.cdf(x - 0.5, loc=0, scale=n_floors/4)
    return prob/prob.sum()

def get_rev_exp_dist(n_floors):
    return np.flip(get_exp_dist(n_floors))

def get_normal_probs(n_floors, mean_floor, variance_floors):
    x = np.arange(1, n_floors+1)
    prob = ss.norm.cdf(x + 0.5, loc=mean_floor, scale=variance_floors)\
                     - ss.norm.cdf(x - 0.5, loc=mean_floor, scale=variance_floors)
    return prob/prob.sum()

def get_normal_dist(n_floors):
    return get_normal_probs(n_floors, n_floors/2 + 0.5, 3)

def get_bimodal_dist(n_floors, peak1, peak2, var):
    p1 = get_normal_probs(n_floors, peak1, var)
    p2 = get_normal_probs(n_floors, peak2, var)
    p = p1 + 4*p2/3
    return p/p.sum()

print("Gaussian distributions")
floors = 8
p = get_normal_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 12
p = get_normal_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 16
p = get_normal_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 20
p = get_normal_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

print("Bimodal distributions")
floors = 8
p = get_bimodal_dist(floors, 2, 7, 1)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)


floors = 12
p = get_bimodal_dist(floors, 3, 9, 2)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)


floors = 16
p = get_bimodal_dist(floors, 3, 13, 3)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)


floors = 20
p = get_bimodal_dist(floors, 4, 15, 3)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)


floors = 20
p = get_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 16
p = get_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 12
p = get_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 8
p = get_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 20
p = get_rev_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 16
p = get_rev_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 12
p = get_rev_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

floors = 8
p = get_rev_exp_dist(floors)
print(floors, " floors:", p, "sum =",p.sum())
_ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)


def get_random_floor(call_probs):
    r = random.uniform(0, 1)
    sum = 0.0
    for floor in range(0, len(call_probs)):
        if r < sum:
            return floor
        sum += call_probs[floor]
    return len(call_probs)
    

# function to find the 2 best floors to park an elevator
# at given distribution of calls at the floor

def get_best_floors(call_probs, iterations):
    distance = [[0 for i in range(len(call_probs))] for j in range(len(call_probs))]
    n_floors = len(call_probs)
    for i in range(iterations):
        floor = get_random_floor(call_probs)
        for i in range(n_floors):
            for j in range(n_floors):
                distance[i][j] += min(abs(floor - i), abs(floor - j))
    min_dist = math.inf
    for i in range(n_floors):
        for j in range(n_floors):
            if distance[i][j] < min_dist:
                min_dist = distance[i][j]
                floor_1 = i
                floor_2 = j
    return floor_1,floor_2

print("Best floors for normal distribution on 8 floors", get_best_floors(get_normal_dist(8), 1000))
print("Best floors for normal distribution on 12 floors", get_best_floors(get_normal_dist(12), 1000))
print("Best floors for normal distribution on 16 floors", get_best_floors(get_normal_dist(16), 1000))
print("Best floors for normal distribution on 20 floors", get_best_floors(get_normal_dist(20), 1000))

print("Best floors for bimodal distribution on 8 floors", get_best_floors(get_bimodal_dist(8, 2, 7, 1), 1000))
print("Best floors for bimodal distribution on 12 floors", get_best_floors(get_bimodal_dist(12, 3, 9, 2), 1000))
print("Best floors for bimodal distribution on 16 floors", get_best_floors(get_bimodal_dist(16, 3, 13, 3), 1000))
print("Best floors for bimodal distribution on 20 floors", get_best_floors(get_bimodal_dist(20, 4, 15, 3), 1000))

print("Best floors for exponential distribution on 8 floors", get_best_floors(get_exp_dist(8), 1000))
print("Best floors for exponential distribution on 12 floors", get_best_floors(get_exp_dist(12), 1000))
print("Best floors for exponential distribution on 16 floors", get_best_floors(get_exp_dist(16), 1000))
print("Best floors for exponential distribution on 20 floors", get_best_floors(get_exp_dist(20), 1000))

print("Best floors for reverse exponential distribution on 8 floors", get_best_floors(get_rev_exp_dist(8), 1000))
print("Best floors for reverse exponential distribution on 12 floors", get_best_floors(get_rev_exp_dist(12), 1000))
print("Best floors for reverse exponential distribution on 16 floors", get_best_floors(get_rev_exp_dist(16), 1000))
print("Best floors for reverse exponential distribution on 20 floors", get_best_floors(get_rev_exp_dist(20), 1000))

def opaque_model(call_dist, iterations):
    distance = 0
    el_1, el_2 = get_best_floors(call_dist, iterations)
    for i in range(iterations):
        f = get_random_floor(call_dist)
        distance += min(abs(el_1 - f), abs(el_2 - f))
    return distance/iterations

print(opaque_model(get_normal_dist(8), 1000))
print(opaque_model(get_normal_dist(12), 1000))
print(opaque_model(get_normal_dist(16), 1000))
print(opaque_model(get_normal_dist(20), 1000))

print(opaque_model(get_bimodal_dist(8, 2, 7, 1), 1000))
print(opaque_model(get_bimodal_dist(12, 3, 9, 2), 1000))
print(opaque_model(get_bimodal_dist(16, 3, 13, 3), 1000))
print(opaque_model(get_bimodal_dist(20, 4, 15, 3), 1000))

print(opaque_model(get_exp_dist(8), 1000))
print(opaque_model(get_exp_dist(12), 1000))
print(opaque_model(get_exp_dist(16), 1000))
print(opaque_model(get_exp_dist(20), 1000))

print(opaque_model(get_rev_exp_dist(8), 10000))
print(opaque_model(get_rev_exp_dist(12), 10000))
print(opaque_model(get_rev_exp_dist(16), 10000))
print(opaque_model(get_rev_exp_dist(20), 10000))

class Environment():
    
    def __init__(self, n_elevators, n_floors):
        self.n = 0
        self.n_floors = n_floors
        self.floor = [-1 for i in range(n_floors)]
        self.total_distance = 0
        
    def carry_to(self, elevator, floor):
        penalty = 0
        distance = abs(self.floor[elevator] - floor)
        avg_dist = self.total_distance/self.n if self.n > self.n_floors*3 else 0 
        if distance > avg_dist:
            penalty = 1
        self.n += 1
#        print("floor:", floor, "elevator:", elevator, "elevator floor:", self.floor[elevator], "distance: ", distance)
        self.total_distance += distance
        return penalty
    
    def return_to(self, elevator, floor):
        self.floor[elevator] = floor
        return floor
    
    def print_distance(self):
        print(self.total_distance/self.n)

class LRI_Elevator():
    def __init__(self, el_id, env, n_floors, k_r):
        self.el_id = el_id
        self.env = env
        self.n_floors = n_floors
        self.floor_probs = [random.uniform(0, 1) for i in range(n_floors)]
        self.floor_probs = np.array(self.floor_probs)
        self.floor_probs = self.floor_probs/sum(self.floor_probs)
        self.floor = self.env.return_to(self.el_id, self.find_best_floor())
        self.k_r = k_r
        
    def update_p_values(self, floor, reward):
        # update on reward
        if reward == 0:
            sum = 0.0
            for i in range(self.n_floors):
                if floor != i:
                    # decrease the probabilities for each other floor
                    self.floor_probs[i] = (1- self.k_r)*self.floor_probs[i]
                    sum += self.floor_probs[i]
            self.floor_probs[floor] = 1 - sum
        else:
            # no change when beta == 1
            pass
#        print(self.floor_probs)
        
    def find_best_floor(self):
        best_prob = 0
        best_floor = 0
        for i in range(self.n_floors):
            if self.floor_probs[i] > best_prob:
                best_prob = self.floor_probs[i]
                best_floor = i
#        print("best floor: ", best_floor)
#        print("best prob: ", best_prob)
        return best_floor

    def carry_to(self, floor):
        self.update_p_values(floor, self.env.carry_to(self.el_id, floor))
        self.floor = self.env.return_to(self.el_id, self.find_best_floor())

class ElevatorBank():
    def __init__(self, n_elevators, n_floors, call_probs, k_r):
        self.env = Environment(n_elevators, n_floors)
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.call_probs = call_probs
        self.elevators = [LRI_Elevator(i, self.env, n_floors, k_r) for i in range(n_elevators)]
        
    def get_best_elevator(self, floor):
        best = self.n_floors*3
        for i in range(self.n_elevators):
            if self.elevators[i].floor == floor:
#                print("best elevator: ", i, self.elevators[i].floor)
                return i
            elif abs(i - floor) < abs(best - floor):
                best = i
#        print("best elevator: ", best, "@", self.elevators[best].floor)
        return best

    def print_state(self, floor):
        pass
#        print("Step:", self.env.n, "call:", floor, "total distance:", self.env.total_distance)
        
    def simulate(self, iterations):
#        print(self.elevators[0].floor, self.elevators[0].floor_probs)
#        print(self.elevators[1].floor, self.elevators[1].floor_probs)
        calls = [0 for i in range(self.n_floors)]
        for i in range(iterations):
            floor = get_random_floor(self.call_probs) - 1
            calls[floor] += 1
            self.elevators[self.get_best_elevator(floor)].carry_to(floor)
        print(np.array(calls)/self.call_probs)
#            self.print_state(floor)
#        print(self.elevators[0].floor, self.elevators[0].floor_probs)
#        print(self.elevators[1].floor, self.elevators[1].floor_probs)

total = 0
for i in range(1):
    bank = ElevatorBank(1, 20, get_normal_dist(20), 0.1)
    bank.simulate(10000000)
    total += bank.env.total_distance/1000
print(total/10000)
#bank.env.print_distance()

print(bank.elevators[0].floor, bank.elevators[0].floor_probs)
print(bank.elevators[1].floor, bank.elevators[1].floor_probs)

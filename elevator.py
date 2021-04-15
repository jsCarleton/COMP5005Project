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

# print sample results for distributions
floors_list = [8, 12, 16, 20]
print("Gaussian distributions")
for floors in floors_list:
    p = get_normal_dist(floors)
    print(floors, " floors:", p, "sum =", p.sum())
    _ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

print("Bimodal distributions")
bimodal_params = [[8, 2, 7, 1], [12, 3, 9, 2], [16, 3, 13, 3], [20, 4, 15, 3]]
for b in bimodal_params:
    p = get_bimodal_dist(b[0], b[1], b[2], b[3])
    print(b[0], " floors:", p, "sum =", p.sum())
    _ = plt.hist(np.random.choice(np.arange(1, b[0]+1), size = 10000, p=p), bins=b[0])

print("Exponential distributions")
for floors in floors_list:
    p = get_exp_dist(floors)
    print(floors, " floors:", p, "sum =", p.sum())
    _ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

print("Reverse exponential distributions")
for floors in floors_list:
    p = get_rev_exp_dist(floors)
    print(floors, " floors:", p, "sum =", p.sum())
    _ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)
print()

# function to return a random floor given a probability distribution
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

print("Best waiting floors")
print("-------------------")
for floors in floors_list:
    print("Normal,", floors, "floors:", get_best_floors(get_normal_dist(floors), 1000))

for b in bimodal_params:
    print("Bimodal,", b[0], "floors:", get_best_floors(get_bimodal_dist(b[0], b[1], b[2], b[3]), 1000))

for floors in floors_list:
    print("Exponential,", floors, "floors:", get_best_floors(get_exp_dist(floors), 1000))

for floors in floors_list:
    print("Reverse exponential,", floors, "floors:", get_best_floors(get_rev_exp_dist(floors), 1000))
print()

# a function that returns the average distance travelled when the distribution is known
def opaque_model(call_dist, iterations):
    distance = 0
    el_1, el_2 = get_best_floors(call_dist, iterations)
    for i in range(iterations):
        f = get_random_floor(call_dist)
        distance += min(abs(el_1 - f), abs(el_2 - f))
    return distance/iterations

print("AWT for opaque models")
print("---------------------")
for floors in floors_list:
    print("Normal,", floors, "floors:", opaque_model(get_normal_dist(floors), 1000))

for b in bimodal_params:
    print("Bimodal,", b[0], "floors:", opaque_model(get_bimodal_dist(b[0], b[1], b[2], b[3]), 1000))

for floors in floors_list:
    print("Exponential,", floors, "floors:", opaque_model(get_exp_dist(floors), 1000))

for floors in floors_list:
    print("Reverse exponential,", floors, "floors:", opaque_model(get_rev_exp_dist(floors), 1000))
print()

class Environment():
    
    def __init__(self, call_probs, dest_probs, n_elevators, n_floors):
        self.n = 0
        self.call_probs = call_probs
        self.dest_probs = dest_probs
        self.n_floors = n_floors
        self.floor = [-1 for i in range(n_floors)]
        self.total_distance = 0

    def get_call_floor(self):
       return get_random_floor(self.call_probs) - 1

    def get_dest_floor(self, call_floor):
        xdest_probs = self.dest_probs.copy()
        xdest_probs[call_floor] = 0
        xdest_probs = xdest_probs/sum(xdest_probs)
        dest_floor = get_random_floor(xdest_probs)
        if dest_floor == call_floor:
            print("whoops!")
        return dest_floor - 1

    def carry_to(self, elevator, call_floor, dest_floor, rest_floor):
        penalty = 0
        distance = abs(self.floor[elevator] - call_floor) + abs(dest_floor - rest_floor)/2
        avg_dist = self.total_distance/self.n if self.n > self.n_floors*3 else 0 
        if distance > avg_dist:
            penalty = 1
        self.n += 1
#        print("floor:", floor, "elevator:", elevator, "elevator floor:", self.floor[elevator], "distance: ", distance)
        self.total_distance += distance
        self.floor[elevator] = rest_floor
        return penalty

    def set_floor(self, elevator, floor):
        self.floor[elevator] = floor
    
    def print_distance(self):
        print(self.total_distance/self.n)

class DoNothing_Elevator():
    def __init__(self, el_id, env, n_floors, k_r):
        self.el_id = el_id
        self.env = env
        self.floor = n_floors/2
        env.set_floor(el_id, self.floor)

    def carry_to(self, call_floor, dest_floor):
        self.env.carry_to(self.el_id, call_floor, dest_floor, dest_floor)
        self.floor = dest_floor

class LRI_Elevator():
    def __init__(self, el_id, env, n_floors, k_r):
        self.el_id = el_id
        self.env = env
        self.n_floors = n_floors
        self.floor_probs = [random.uniform(0, 1) for i in range(n_floors)]
        self.floor_probs = np.array(self.floor_probs)
        self.floor_probs = self.floor_probs/sum(self.floor_probs)
        self.floor = self.env.set_floor(self.el_id, self.find_best_floor())
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

    def carry_to(self, call_floor, dest_floor):
        self.update_p_values(floor, self.env.carry_to(self.el_id, call_floor, dest_floor, self.find_best_floor()))

class ElevatorBank():
    def __init__(self, env, elevator_class, n_elevators, n_floors, k_r):
        self.env = env
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.elevators = [elevator_class(i, self.env, n_floors, k_r) for i in range(n_elevators)]
        
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
        dests = [0 for i in range(self.n_floors)]
        for i in range(iterations):
            call_floor = self.env.get_call_floor()
            calls[call_floor] += 1
            dest_floor = self.env.get_dest_floor(call_floor)
            dests[dest_floor] += 1
            self.elevators[self.get_best_elevator(call_floor)].carry_to(call_floor, dest_floor)
#        print(np.array(calls)/self.call_probs)
#            self.print_state(floor)
#        print(self.elevators[0].floor, self.elevators[0].floor_probs)
#        print(self.elevators[1].floor, self.elevators[1].floor_probs)

elevators = 1
iterations = 20000
print("AWT for do-nothing models,", elevators, "elevators")
print("--------------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", bank.env.total_distance/iterations)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", bank.env.total_distance/iterations)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", bank.env.total_distance/iterations)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", bank.env.total_distance/iterations)
print()

print("AWT for L-RI models,", elevators, "elevators")
print("--------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", bank.env.total_distance/iterations)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", bank.env.total_distance/iterations)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", bank.env.total_distance/iterations)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", bank.env.total_distance/iterations)
print()


#total = 0
#for i in range(1):
#    bank = ElevatorBank(1, 20, get_normal_dist(20), 0.1)
#    bank.simulate(10000000)
#    total += bank.env.total_distance/1000
#print(total/10000)
#bank.env.print_distance()

#print(bank.elevators[0].floor, bank.elevators[0].floor_probs)
#print(bank.elevators[1].floor, bank.elevators[1].floor_probs)

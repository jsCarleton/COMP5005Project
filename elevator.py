from scipy.stats import norm
from scipy.stats import expon
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import random
import math

print(np.__version__)

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
    total = 0.0
    for floor in range(len(call_probs)):
        if r < total:
            return floor - 1
        total += call_probs[floor]
    return len(call_probs) - 1    

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
        self.total_distance = 0
        self.penalties = 0

    def reset(self):
        self.total_distance = 0
        
    def get_call_floor(self):
       return get_random_floor(self.call_probs)

    def get_dest_floor(self, call_floor):
        xdest_probs = self.dest_probs.copy()
        xdest_probs[call_floor] = 0
        xdest_probs = xdest_probs/xdest_probs.sum()
        dest_floor = get_random_floor(xdest_probs)
        if dest_floor == call_floor:
            print("whoops!", dest_floor, call_floor, len(self.call_probs))
        return dest_floor

    def carry_to(self, elevator, call_floor, dest_floor, rest_floor):
        penalty = 0
        distance = abs(elevator.floor - call_floor) + abs(dest_floor - rest_floor)/2
        avg_dist = self.total_distance/self.n if self.n > 0 else 0 
        if distance > avg_dist:
            penalty = 1
            self.penalties += 1
        self.n += 1
#        print("floor:", floor, "elevator:", elevator, "elevator floor:", self.floor[elevator], "distance: ", distance)
        self.total_distance += distance
#        elevator.floor = rest_floor
        return penalty

    def get_best_floor(self):
        best_prob = 0
        best_floor = 0
        for i in range(self.n_floors):
            if self.call_probs[i] > best_prob:
                best_floor = i
                best_prob = self.call_probs[i]
        return best_floor
    
    def print_distance(self):
        print(self.total_distance/self.n)

class Oracle_Elevator():
    def __init__(self, env, n_floors, k_r):
        self.env = env
        self.floor = n_floors/2
        self.rest_floors = [0 for i in range(n_floors)]

    def carry_to(self, call_floor, dest_floor):
        self.env.carry_to(self, call_floor, dest_floor, env.get_best_floor())
        self.floor = env.get_best_floor()
        self.rest_floors[self.floor] += 1

    def get_best_floor(self):
        return env.get_best_floor()

class DoNothing_Elevator():
    def __init__(self, env, n_floors, k_r):
        self.env = env
        self.floor = n_floors/2

    def carry_to(self, call_floor, dest_floor):
        self.env.carry_to(self, call_floor, dest_floor, dest_floor)
        self.floor = dest_floor

    def get_best_floor(self):
        return self.floor

class LRI_Elevator():
    def __init__(self, env, n_floors, k_r):
        self.env = env
        self.n_floors = n_floors
        self.floor_probs = [random.uniform(0, 1) for i in range(n_floors)]
        self.floor_probs = np.array(self.floor_probs)
        self.floor_probs = self.floor_probs/sum(self.floor_probs)
        self.floor_probs = [1/n_floors for i in range(n_floors)]
        self.k_r = k_r
        self.floor = n_floors//2
        self.rest_floor = self.floor
        self.penalties = 0
        self.rest_floors = [0 for i in range(n_floors)]

    def update_state(self, floor, penalty):
        # update on reward
        if penalty == 0:
            total = 0.0
            for i in range(self.n_floors):
                if floor != i:
                    # decrease the probabilities for each other floor
                    self.floor_probs[i] = (1 - self.k_r/self.n_floors)*self.floor_probs[i]        
                    total += self.floor_probs[i]
            self.floor_probs[floor] = 1 - total
            if abs(np.array(self.floor_probs).sum() - 1.0) > 0.95:
                print("waitaminute!")
        else:
            # no change when beta == 1
            self.penalties += 1
#            pass
#        print(self.floor_probs)

    def get_best_floor(self):
        best_floor = 0
        best_prob = 0
        for i in range(self.n_floors):
            if self.floor_probs[i] > best_prob:
                best_floor = i
                best_prob = self.floor_probs[i]
        return best_floor
        
    def set_rest_floor(self):
        self.rest_floor = self.get_best_floor()
#        print("best floor: ", best_floor)
#        print("best prob: ", best_prob)

    def carry_to(self, call_floor, dest_floor):
        # tell the environment what we're doing, see if it rewards us
        self.set_rest_floor()
        reward = self.env.carry_to(self, call_floor, dest_floor, self.rest_floor)
        self.update_state(self.floor, reward)
        self.floor = self.rest_floor
        self.rest_floors[self.rest_floor] += 1

class Pursuit_Elevator():
    def __init__(self, env, n_floors, k_r):
        self.env = env
        self.n_floors = n_floors
        self.floor_probs = [random.uniform(0, 1) for i in range(n_floors)]
        self.floor_probs = np.array(self.floor_probs)
        self.floor_probs = self.floor_probs/sum(self.floor_probs)
        self.floor_probs = np.array([1/n_floors for i in range(n_floors)])
        self.k_r = k_r
        self.floor = n_floors//2
        self.rest_floor = self.floor
        self.penalties = 0
        self.rest_floors = [0 for i in range(n_floors)]
        self.rewards = [0 for i in range(n_floors)]
        self.attempts = [0 for i in range(n_floors)]
        my_floor = int(n_floors*random.uniform(0, 1))
        for i in range(1000):
            rest_floor = int(n_floors*random.uniform(0, 1))
            call_floor = int(n_floors*random.uniform(0, 1))
            dest_floor = int(n_floors*random.uniform(0, 1))
            self.update_state(my_floor, self.env.carry_to(self, rest_floor,\
                call_floor, rest_floor))
            my_floor = rest_floor
        self.env.reset()

    def update_state(self, floor, penalty):
        # update the rewards, attempts data
        self.attempts[floor] += 1
        self.rewards[floor] += (1 - penalty)
        # find the most successful floor
        best_floor = 0
        best_reward_rate = 0
        for i in range(self.n_floors):
            if self.attempts[i] > 0 and self.rewards[i]/self.attempts[i] > best_reward_rate:
                best_floor = i
                best_reward_rate = self.rewards[i]/self.attempts[i]
        # create the unit vector for that floor
        e_b = np.array([1 if i == best_floor else 0 for i in range(self.n_floors)])
        # update the p values
        self.floor_probs = (1 - self.k_r)*self.floor_probs + self.k_r*e_b
        
    def get_best_floor(self):
        best_floor = 0
        best_prob = 0
        for i in range(self.n_floors):
            if self.floor_probs[i] > best_prob:
                best_floor = i
                best_prob = self.floor_probs[i]
        return best_floor
        
    def set_rest_floor(self):
        self.rest_floor = self.get_best_floor()
#        print("best floor: ", best_floor)
#        print("best prob: ", best_prob)

    def carry_to(self, call_floor, dest_floor):
        # tell the environment what we're doing, see if it rewards us
        self.set_rest_floor()
        reward = self.env.carry_to(self, call_floor, dest_floor, self.rest_floor)
        self.update_state(self.floor, reward)
        self.floor = self.rest_floor
        self.rest_floors[self.rest_floor] += 1


class ElevatorBank():
    def __init__(self, env, elevator_class, n_elevators, n_floors, k_r):
        self.env = env
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.elevators = [elevator_class(env, n_floors, k_r) for i in range(n_elevators)]
        
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
        self.calls = [0 for i in range(self.n_floors)]
        self.dests = [0 for i in range(self.n_floors)]
        for i in range(iterations):
            # get the call floor
            call_floor = self.env.get_call_floor()
            # get the destination floor, making sure it's different from the call floor
            self.calls[call_floor] += 1
            dest_floor = self.env.get_dest_floor(call_floor)
            self.dests[dest_floor] += 1
            # tell the elevator closest to the call floor to handle this
            self.elevators[self.get_best_elevator(call_floor)].carry_to(call_floor, dest_floor)
#        print(np.array(calls)/self.call_probs)
#            self.print_state(floor)
#        print(self.elevators[0].floor, self.elevators[0].floor_probs)
#        print(self.elevators[1].floor, self.elevators[1].floor_probs)

elevators = 1
iterations = 100
print("AWT for Oracle models,", elevators, "elevators")
print("----------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Oracle_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
#    print("Penalties:", env.penalties)
#    print("Calls:", bank.calls)
#    print("Rest floors:", bank.elevators[0].rest_floors)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, b[0])
    bank = ElevatorBank(env, Oracle_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Oracle_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Oracle_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)
print()

print("AWT for do-nothing models,", elevators, "elevators")
print("--------------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, b[0])
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, DoNothing_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print("Penalties:", env.penalties)
 #   print("Calls:", bank.calls)
print()

print("AWT for L-RI models,", elevators, "elevators")
print("--------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
#    print(bank.elevators[0].floor_probs)
#    print("Penalties:", env.penalties)
#    print("ePenalties:", bank.elevators[0].penalties)
#    print("Calls:", bank.calls)
#    print("Rest floors:", bank.elevators[0].rest_floors)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, b[0])
    bank = ElevatorBank(env, LRI_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print(bank.elevators[0].floor_probs)
 #   print("Penalties:", env.penalties)
 #   print("ePenalties:", bank.elevators[0].penalties)
 #   print("Calls:", bank.calls)
 #   print("Rest floors:", bank.elevators[0].rest_floors)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print(bank.elevators[0].floor_probs)
 #   print("Penalties:", env.penalties)
 #   print("ePenalties:", bank.elevators[0].penalties)
 #   print("Calls:", bank.calls)
 #   print("Rest floors:", bank.elevators[0].rest_floors)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, LRI_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
  #  print(bank.elevators[0].floor_probs)
  #  print("Penalties:", env.penalties)
  #  print("ePenalties:", bank.elevators[0].penalties)
  #  print("Calls:", bank.calls)
  #  print("Rest floors:", bank.elevators[0].rest_floors)
print()

print("AWT for Pursuit models,", elevators, "elevators")
print("-----------------------------------")
for floors in floors_list:
    env = Environment(get_normal_dist(floors), get_normal_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Pursuit_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Normal,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
#    print(bank.elevators[0].floor_probs)
#    print(bank.elevators[0].rewards)
#    print(bank.elevators[0].attempts)
#    print("Penalties:", env.penalties)
#    print("ePenalties:", bank.elevators[0].penalties)
#    print("Calls:", bank.calls)
#    print("Rest floors:", bank.elevators[0].rest_floors)

for b in bimodal_params:
    env = Environment(get_bimodal_dist(b[0], b[1], b[2], b[3]), get_bimodal_dist(b[0], b[1], b[2], b[3]), elevators, b[0])
    bank = ElevatorBank(env, Pursuit_Elevator, elevators, b[0],  0.1)
    bank.simulate(iterations)
    print("Bimodal,", b[0], "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print(bank.elevators[0].floor_probs)
 #   print("Penalties:", env.penalties)
 #   print("ePenalties:", bank.elevators[0].penalties)
 #   print("Calls:", bank.calls)
 #   print("Rest floors:", bank.elevators[0].rest_floors)

for floors in floors_list:
    env = Environment(get_exp_dist(floors), get_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Pursuit_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
 #   print(bank.elevators[0].floor_probs)
 #   print("Penalties:", env.penalties)
 #   print("ePenalties:", bank.elevators[0].penalties)
 #   print("Calls:", bank.calls)
 #   print("Rest floors:", bank.elevators[0].rest_floors)

for floors in floors_list:
    env = Environment(get_rev_exp_dist(floors), get_rev_exp_dist(floors), elevators, floors)
    bank = ElevatorBank(env, Pursuit_Elevator, elevators, floors, 0.1)
    bank.simulate(iterations)
    print("Reverse exponential,", floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
  #  print(bank.elevators[0].floor_probs)
  #  print("Penalties:", env.penalties)
  #  print("ePenalties:", bank.elevators[0].penalties)
  #  print("Calls:", bank.calls)
  #  print("Rest floors:", bank.elevators[0].rest_floors)
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

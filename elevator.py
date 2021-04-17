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

# frozen distributions that we use for testing purposes
normal_dist = {8: np.array([0.08249252, 0.11477478, 0.14304307, 0.15968963, 0.15968963, 0.14304307,
 0.11477478, 0.08249252]),
 12: np.array([0.02623387, 0.04549071, 0.07065904, 0.09831043, 0.12252366, 0.13678229,
 0.13678229, 0.12252366, 0.09831043, 0.07065904, 0.04549071, 0.02623387]),
 16:  np.array([0.00603115, 0.01303466, 0.02523353, 0.04375607, 0.0679647,  0.0945617,
 0.11785164, 0.13156656, 0.13156656, 0.11785164, 0.0945617,  0.0679647,
 0.04375607, 0.02523353, 0.01303466, 0.00603115]),
20:  np.array([0.00092163, 0.00248261, 0.00599009, 0.01294591, 0.02506173, 0.04345816,
 0.06750196, 0.09391788, 0.11704925, 0.13067079, 0.13067079, 0.11704925,
 0.09391788, 0.06750196, 0.04345816, 0.02506173, 0.01294591, 0.00599009,
 0.00248261, 0.00092163])}
bimodal_dist = {8:  np.array([0.11101535, 0.17586142, 0.11115571, 0.03148955, 0.03985109, 0.14812573,
 0.23448069, 0.14802046]),
12: np.array([0.05801863, 0.08399082, 0.09603029, 0.08920317, 0.07454183, 0.07047585,
 0.08535406, 0.10840567, 0.11866747, 0.10420975, 0.07205295, 0.03904951]),
16: np.array([0.05709516, 0.06741996, 0.07146879, 0.06830664, 0.05960559, 0.04913846,
 0.04134911, 0.03971162, 0.0454987,  0.05726739, 0.07119836, 0.08232936,
 0.08640746, 0.08159712, 0.06912519, 0.05248108]),
20: np.array([0.03935687, 0.05183086, 0.06115673, 0.06468768, 0.06144675, 0.05272956,
 0.04166456, 0.03204442, 0.0270987,  0.02866602, 0.03678682, 0.0495866,
 0.06353425, 0.07432238, 0.07833311, 0.07408299, 0.06279247, 0.04768184,
 0.03243443, 0.01976293])}
exp_dist = {8: np.array([0.40081044, 0.24310382, 0.14744992, 0.0894329, 0.05424379, 0.03290052,
 0.01995518, 0.01210343]),
12: np.array([0.28875747, 0.20690377, 0.14825303, 0.10622794, 0.07611564, 0.05453924,
 0.03907907, 0.02800138, 0.02006387, 0.01437639, 0.01030113, 0.00738108]),
16: np.array([0.22532621, 0.17548423, 0.13666726, 0.10643657, 0.08289288, 0.06455704,
 0.05027707, 0.03915582, 0.03049459, 0.02374921, 0.0184959,  0.01440462,
 0.01121833, 0.00873685, 0.00680426, 0.00529916]),
20: np.array([0.18465125, 0.15117966, 0.12377544, 0.10133876, 0.08296916, 0.0679294,
 0.05561589, 0.04553444, 0.03728044, 0.03052265, 0.02498983, 0.02045994,
 0.01675118, 0.01371471, 0.01122865, 0.00919324, 0.00752679, 0.00616242,
 0.00504536, 0.00413079])}
rev_exp_dist = {8: np.array([0.01210343, 0.01995518, 0.03290052, 0.05424379, 0.0894329, 0.14744992,
 0.24310382, 0.40081044]),
12: np.array([0.00738108, 0.01030113, 0.01437639, 0.02006387, 0.02800138, 0.03907907,
 0.05453924, 0.07611564, 0.10622794, 0.14825303, 0.20690377, 0.28875747]),
16: np.array([0.00529916, 0.00680426, 0.00873685, 0.01121833, 0.01440462, 0.0184959,
 0.02374921, 0.03049459, 0.03915582, 0.05027707, 0.06455704, 0.08289288,
 0.10643657, 0.13666726, 0.17548423, 0.22532621]),
20: np.array([0.00413079, 0.00504536, 0.00616242, 0.00752679, 0.00919324, 0.01122865,
 0.01371471, 0.01675118, 0.02045994, 0.02498983, 0.03052265, 0.03728044,
 0.04553444, 0.05561589, 0.0679294,  0.08296916, 0.10133876, 0.12377544,
 0.15117966, 0.18465125])}

# print sample results for distributions
floors_list = [8, 12, 16, 20]
bimodal_params = [[8, 2, 7, 1], [12, 3, 9, 2], [16, 3, 13, 3], [20, 4, 15, 3]]
if 1 == 0:
    print("Gaussian distributions")
    for floors in floors_list:
        p = get_normal_dist(floors)
        print(floors, " floors:", p, "sum =", p.sum())
        _ = plt.hist(np.random.choice(np.arange(1, floors+1), size = 10000, p=p), bins=floors)

    print("Bimodal distributions")
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
elevator_types = {"Oracle": Oracle_Elevator, "Do Nothing": DoNothing_Elevator, "L-RI": LRI_Elevator, "Pursuit": Pursuit_Elevator}
distributions = {"Normal": normal_dist, "Bimodal": bimodal_dist, "Exponential": exp_dist, "Reverse exponential": rev_exp_dist}
for etype in elevator_types:
    print("AWT for", etype, "models,", elevators, "elevators")
    print("----------------------------------")
    for dist in distributions:
        for floors in floors_list:
            env = Environment(distributions[dist][floors], distributions[dist][floors], elevators, floors)
            bank = ElevatorBank(env, elevator_types[etype], elevators, floors, 0.1)
            bank.simulate(iterations)
            print(dist, floors, "floors:", env.total_distance/iterations, bank.elevators[0].get_best_floor())
#           print(bank.elevators[0].floor_probs)
#           print("Penalties:", env.penalties)
#           print("ePenalties:", bank.elevators[0].penalties)
#           print("Calls:", bank.calls)
#           print("Rest floors:", bank.elevators[0].rest_floors)
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

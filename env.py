import numpy as np
from pypower.api import ppoption, runopf
from case18 import case18

class envs(object):
    def __init__(self):
        self.num_station =2
        self.num_EV1_seat = 10
        self.num_EV2_seat = 10
        self.max_EV1_waiting_seat = 10
        self.max_EV2_waiting_seat = 10        
        self.current_empnum_seat_EV1 = 10
        self.current_empnum_seat_EV2 = 10
        self.alpha1 = 20 # Price weight
        self.alpha2 = 120  # Price traveling 
        self.alpha3 = 1  # Price waiting 

        self.Pr_ev1 = 200
        self.Pr_ev2 = 200
        self.EV1_ch_rate_sum_mean = 0
        self.EV2_ch_rate_sum_mean = 0

        self.ppopt = ppoption(PF_ALG=2,  OUT_ALL = 0, VERBOSE=0)
        self.ppc = case18()
        self.busindex = [2, 7]
        self.dem_inj = self.ppc["bus"][:, 2]
        

        self.waiting_car_ev1 = 0
        self.waiting_car_ev2 = 0        

        self.EV1_ch_rate_sum_list = []
        self.EV2_ch_rate_sum_list = [] 

        self.current_res_parking_time_EV1 = np.zeros(self.num_EV1_seat)
        self.current_res_parking_time_EV2 = np.zeros(self.num_EV2_seat)

        self.current_res_demand_EV1 = np.zeros(self.num_EV1_seat)
        self.current_res_demand_EV2 = np.zeros(self.num_EV2_seat)

        self.current_seat_statue_EV1 =   np.zeros(self.num_EV1_seat)  # 1: Charging, 0: empty
        self.current_seat_statue_EV2 =   np.zeros(self.num_EV2_seat)  # 2: Charging, 0: empty

        self.current_seat_price_EV1 = np.zeros(self.num_EV1_seat)
        self.current_seat_price_EV2 = np.zeros(self.num_EV2_seat)

        self.waiting_seat_EV1 = np.zeros(self.max_EV1_waiting_seat) # 1: waiting, 0: empty
        self.waiting_seat_EV2 = np.zeros(self.max_EV2_waiting_seat) # 1: waiting, 0: empty

        self.waiting_seat_EV1_price = np.zeros(self.max_EV1_waiting_seat) # 1: waiting, 0: empty
        self.waiting_seat_EV2_price = np.zeros(self.max_EV2_waiting_seat) # 1: waiting, 0: empty        

        self.waiting_seat_EV1_demand = np.zeros(self.max_EV1_waiting_seat) # 1: waiting, 0: empty
        self.waiting_seat_EV2_demand = np.zeros(self.max_EV2_waiting_seat) # 1: waiting, 0: empty 

        self.state_ev1 = np.concatenate((self.current_res_parking_time_EV1, self.current_res_demand_EV1, self.current_seat_statue_EV1, self.current_seat_price_EV1, \
                                         self.waiting_seat_EV1, self.waiting_seat_EV1_price, self.waiting_seat_EV1_demand), axis=0) 
        self.state_ev2 = np.concatenate((self.current_res_parking_time_EV1, self.current_res_demand_EV1, self.current_seat_statue_EV1, self.current_seat_price_EV1, \
                                         self.waiting_seat_EV1, self.waiting_seat_EV1_price, self.waiting_seat_EV1_demand), axis=0)
        self.states = np.vstack((self.state_ev1, self.state_ev2)) 

       

    def step(self, actions, t):
        action_EV1 = actions[0]
        action_EV2 = actions[1]
        EV1_price = 0.35 * action_EV1[0] + 0.35 # [-1, 1] -> [0, 0.8]


        EV1_ch_rate = 3 * action_EV1[1:] + 7 # [-1, 1] -> [4, 10]

        EV2_price = 0.35 * action_EV2[0] + 0.35 # [-1, 1] -> [0, 0.8]

        EV2_ch_rate = 3 * action_EV2[1:] + 7 # [-1, 1] -> [4, 10]
        self.iter = t



        self.choose_station(EV1_price, EV2_price)

        self.update_already_cars(EV1_ch_rate, EV2_ch_rate) # The already charged car's Res time and demand. Updated current seat statue.
        self.update_waiting_cars()
        self.update_new_arrival_car(self.EV1_arrival_num, self.EV2_arrival_num, EV1_price, EV2_price) # The new arrival    Updated current seat statue. 

        rewardEV1, rewardEV2, Pr_ev1, Pr_ev2, EV_load1, EV_load2 = self.update_current_reward(EV1_ch_rate, EV2_ch_rate)
        self.state_ev1 = np.concatenate((self.current_res_parking_time_EV1, self.current_res_demand_EV1, self.current_seat_statue_EV1, self.current_seat_price_EV1, \
                                         self.waiting_seat_EV1, self.waiting_seat_EV1_price, self.waiting_seat_EV1_demand), axis=0)

        self.state_ev2 = np.concatenate((self.current_res_parking_time_EV2, self.current_res_demand_EV2, self.current_seat_statue_EV2, self.current_seat_price_EV2, \
                                         self.waiting_seat_EV2, self.waiting_seat_EV2_price, self.waiting_seat_EV2_demand), axis=0)
        
        states = np.vstack((self.state_ev1, self.state_ev2)) 
        rewards = np.vstack((rewardEV1, rewardEV2))
        Pr = np.vstack((Pr_ev1, Pr_ev2))
        load = np.vstack((EV_load1, EV_load2))
        
        return rewards, states, Pr, load


        # state includes   current_seat_statue, current_res_demand, current_res_parking_time and waiting car 

    def choose_station(self, EV1_price, EV2_price):
        driver_num = np.random.poisson(6)             ## Each Time there are 4-6 EVs per hour 
        driver_hr1  = np.random.uniform(0, 1, size=driver_num)     ## Each EV travelling time is 0-3 Hours
        driver_hr2  = np.random.uniform(0, 1, size=driver_num)     ## Each EV travelling time is 0-3 Hours
        if int(np.sum(self.waiting_seat_EV1)) <= 9:
            waiting_EV1_time = np.sort(self.current_res_parking_time_EV1)[int(np.sum(self.waiting_seat_EV1))-1]
        else:
            waiting_EV1_time = 1e6

        if int(np.sum(self.waiting_seat_EV1)) <= 9:
            waiting_EV2_time = np.sort(self.current_res_parking_time_EV2)[int(np.sum(self.waiting_seat_EV2))-1]
        else:
            waiting_EV2_time = 1e6

        driver_choice1 = self.alpha1 * EV1_price + self.alpha2 * driver_hr1 + self.alpha3 * waiting_EV1_time 
        driver_choice2 = self.alpha1 * EV2_price + self.alpha2 * driver_hr2 + self.alpha3 * waiting_EV2_time 

        driver_choice = np.vstack((driver_choice1 , driver_choice2)).argmin(axis=0)

        EV1_arrival_num = (driver_choice==0).sum()
        EV2_arrival_num = (driver_choice==1).sum()

        self.EV1_arrival_num = EV1_arrival_num
        self.EV2_arrival_num = EV2_arrival_num



    def demand_price(self, price):
        if price <= 0.7:
            demand = 80 - 100 * price
        else:
            demand = 0
        return demand

    def demand_price2(self, price, behaviors):
        demands = []
        for behavior in behaviors:
            if behavior == 0:
                demands.append(80 - 100 * price if price <= 0.7 else 0)
            elif behavior == 1:
                demands.append(60)
            else:
                demands.append((100 - 80 * price) if price <= 0.7 else 0)
                
        return demands

    def update_already_cars(self, EV1_ch_rate, EV2_ch_rate):

        self.current_res_parking_time_EV1 =  np.maximum(0, self.current_res_parking_time_EV1 - 1)
        self.current_res_parking_time_EV2 =  np.maximum(0, self.current_res_parking_time_EV2 - 1)

        self.current_res_demand_EV1 = np.maximum(0, self.current_res_demand_EV1 - EV1_ch_rate * self.current_seat_statue_EV1)
        self.current_res_demand_EV2 = np.maximum(0, self.current_res_demand_EV2 - EV2_ch_rate * self.current_seat_statue_EV2)
 

        num_EV1_leaving = 0
        for i in range(self.num_EV1_seat):
            if self.current_res_parking_time_EV1[i] == 0 or self.current_res_demand_EV1[i] == 0:
                self.current_seat_statue_EV1[i] = 0
                self.current_seat_price_EV1[i] = 0
                self.current_res_parking_time_EV1[i] = 0
                self.current_res_demand_EV1[i] = 0
                num_EV1_leaving +=1


        num_EV2_leaving = 0
        for i in range(self.num_EV2_seat):
            if self.current_res_parking_time_EV2[i] == 0 or self.current_res_demand_EV2[i] == 0:
                self.current_seat_statue_EV2[i] = 0
                self.current_seat_price_EV2[i] = 0
                self.current_res_parking_time_EV2[i] = 0
                self.current_res_demand_EV2[i] = 0
                num_EV2_leaving += 1

    
    def update_waiting_cars(self):
        self.current_empnum_seat_EV1 = self.num_EV1_seat - self.current_seat_statue_EV1.sum() 
        waiting_seat_EV1_num = np.sum(self.waiting_seat_EV1) # number of waiting car

        if self.current_empnum_seat_EV1 >= waiting_seat_EV1_num:
            emptyseatev1 = np.where(self.current_seat_statue_EV1 == 0)[0]
            waitingseatev1 = np.where(self.waiting_seat_EV1 == 1)[0]

            self.current_seat_statue_EV1[emptyseatev1[:int(waiting_seat_EV1_num)]] = 1
            self.current_res_demand_EV1[emptyseatev1[:int(waiting_seat_EV1_num)]] = self.waiting_seat_EV1_demand[waitingseatev1]
            self.current_seat_price_EV1[emptyseatev1[:int(waiting_seat_EV1_num)]] = self.waiting_seat_EV1_price[waitingseatev1]
            
            self.current_res_parking_time_EV1[emptyseatev1[:int(waiting_seat_EV1_num)]] = np.random.uniform(1, 3, size=int(waiting_seat_EV1_num)) 

            self.waiting_seat_EV1[waitingseatev1] = 0
            self.waiting_seat_EV1_demand[waitingseatev1] = 0
            self.waiting_seat_EV1_price[waitingseatev1] = 0
        else:
            emptyseatev1 = np.where(self.current_seat_statue_EV1 == 0)[0]
            waitingseatev1 = np.where(self.waiting_seat_EV1 == 1)[0]
            self.current_seat_statue_EV1[emptyseatev1] = 1
            self.current_res_demand_EV1[emptyseatev1] = self.waiting_seat_EV1_demand[waitingseatev1[:int(self.current_empnum_seat_EV1)]]
            self.current_seat_price_EV1[emptyseatev1] = self.waiting_seat_EV1_price[waitingseatev1[:int(self.current_empnum_seat_EV1)]]
            self.current_res_parking_time_EV1[emptyseatev1] = np.random.uniform(1, 3, size=int(self.current_empnum_seat_EV1)) 

            self.waiting_seat_EV1[waitingseatev1[:int(self.current_empnum_seat_EV1)]] = 0
            self.waiting_seat_EV1_demand[waitingseatev1[:int(self.current_empnum_seat_EV1)]] = 0
            self.waiting_seat_EV1_price[waitingseatev1[:int(self.current_empnum_seat_EV1)]] = 0            


        self.current_empnum_seat_EV2 = self.num_EV2_seat - self.current_seat_statue_EV2.sum() 
        waiting_seat_EV2_num = np.sum(self.waiting_seat_EV2)

        if self.current_empnum_seat_EV2 >= waiting_seat_EV2_num:
            emptyseatev2 = np.where(self.current_seat_statue_EV2 == 0)[0]
            waitingseatev2 = np.where(self.waiting_seat_EV2 == 1)[0]
            self.current_seat_statue_EV2[emptyseatev2[:int(waiting_seat_EV2_num)]] = 1
            self.current_res_demand_EV2[emptyseatev2[:int(waiting_seat_EV2_num)]] = self.waiting_seat_EV2_demand[waitingseatev2]
            self.current_seat_price_EV2[emptyseatev2[:int(waiting_seat_EV2_num)]] = self.waiting_seat_EV2_price[waitingseatev2]
            self.current_res_parking_time_EV2[emptyseatev2[:int(waiting_seat_EV2_num)]] = np.random.uniform(1, 3, size=int(waiting_seat_EV2_num)) 

            self.waiting_seat_EV2[waitingseatev2] = 0
            self.waiting_seat_EV2_demand[waitingseatev2] = 0
            self.waiting_seat_EV2_price[waitingseatev2] = 0
        else:
            emptyseatev2 = np.where(self.current_seat_statue_EV2 == 0)[0]
            waitingseatev2 = np.where(self.waiting_seat_EV2 == 1)[0]
            self.current_seat_statue_EV2[emptyseatev2] = 1
            self.current_res_demand_EV2[emptyseatev2] = self.waiting_seat_EV2_demand[waitingseatev2[:int(self.current_empnum_seat_EV2)]]
            self.current_seat_price_EV2[emptyseatev2] = self.waiting_seat_EV2_price[waitingseatev2[:int(self.current_empnum_seat_EV2)]]
            self.current_res_parking_time_EV2[emptyseatev2] = np.random.uniform(1, 3, size=int(self.current_empnum_seat_EV2)) 

            self.waiting_seat_EV2[waitingseatev2[:int(self.current_empnum_seat_EV2)]] = 0
            self.waiting_seat_EV2_demand[waitingseatev2[:int(self.current_empnum_seat_EV2)]] = 0
            self.waiting_seat_EV2_price[waitingseatev2[:int(self.current_empnum_seat_EV2)]] = 0         



    def update_new_arrival_car(self, EV1_arrival_num, EV2_arrival_num, EV1_price, EV2_price):
        self.current_empnum_seat_EV1 = self.num_EV1_seat - self.current_seat_statue_EV1.sum() 
    
        if self.current_empnum_seat_EV1 >= EV1_arrival_num:
            emptyseatev1 = np.where(self.current_seat_statue_EV1 == 0)[0]
            self.current_seat_statue_EV1[emptyseatev1[:int(EV1_arrival_num)]] = 1
            behavior_statuses = [np.random.randint(0, 3) for _ in range(EV1_arrival_num)]
            self.current_res_demand_EV1[emptyseatev1[:int(EV1_arrival_num)]] = self.demand_price2(EV1_price, behavior_statuses)
            self.current_seat_price_EV1[emptyseatev1[:int(EV1_arrival_num)]] = EV1_price
            self.current_res_parking_time_EV1[emptyseatev1[:int(EV1_arrival_num)]] = np.random.uniform(1, 3, size=int(EV1_arrival_num)) # Initialize the parking time

        else:
            emptyseatev1 = np.where(self.current_seat_statue_EV1 == 0)[0]
            self.current_seat_statue_EV1[emptyseatev1] =1
            behavior_statuses = [np.random.randint(0, 3) for _ in range(len(emptyseatev1))]
            self.current_res_demand_EV1[emptyseatev1] = self.demand_price2(EV1_price, behavior_statuses)
            self.current_seat_price_EV1[emptyseatev1] = EV1_price
            waiting_car_ev1_num = EV1_arrival_num - self.current_empnum_seat_EV1
            
            self.waiting_seat_EV1_num = np.sum(self.waiting_seat_EV1)
            if self.waiting_seat_EV1_num >= waiting_car_ev1_num:
                empty_waiting_seat_ev1 = np.where(self.waiting_seat_EV1 == 0)[0]
                self.waiting_seat_EV1[empty_waiting_seat_ev1[:int(waiting_car_ev1_num)]] = 1
                chosen = len(empty_waiting_seat_ev1[:int(waiting_car_ev1_num)])
                behavior_statuses = [np.random.randint(0, 3) for _ in range(chosen)]
                self.waiting_seat_EV1_demand[empty_waiting_seat_ev1[:int(waiting_car_ev1_num)]] = self.demand_price2(EV1_price, behavior_statuses)
                self.waiting_seat_EV1_price[empty_waiting_seat_ev1[:int(waiting_car_ev1_num)]] = EV1_price
            else:
                empty_waiting_seat_ev1 = np.where(self.waiting_seat_EV1 == 0)[0]
                self.waiting_seat_EV1[empty_waiting_seat_ev1] = 1
                behavior_statuses = [np.random.randint(0, 3) for _ in range(len(empty_waiting_seat_ev1))]
                self.waiting_seat_EV1_demand[empty_waiting_seat_ev1] = self.demand_price2(EV1_price, behavior_statuses)
                self.waiting_seat_EV1_price[empty_waiting_seat_ev1] = EV1_price
 


        self.current_empnum_seat_EV2 = self.num_EV2_seat - self.current_seat_statue_EV2.sum()
        if self.current_empnum_seat_EV2 >= EV2_arrival_num:
            emptyseatev2 = np.where(self.current_seat_statue_EV2 == 0)[0]
            self.current_seat_statue_EV2[emptyseatev2[:int(EV2_arrival_num)]] =1
            behavior_statuses = [np.random.randint(0, 3) for _ in range(EV2_arrival_num)]
            self.current_res_demand_EV2[emptyseatev2[:int(EV2_arrival_num)]] = self.demand_price2(EV2_price, behavior_statuses)
            self.current_seat_price_EV2[emptyseatev2[:int(EV2_arrival_num)]] = EV2_price
            self.current_res_parking_time_EV2[emptyseatev2[:int(EV2_arrival_num)]] = np.random.uniform(1, 3, size=int(EV2_arrival_num)) # Initialize the parking time

        else:
            emptyseatev2 = np.where(self.current_seat_statue_EV2 == 0)[0]
            self.current_seat_statue_EV2[emptyseatev2] =1
            behavior_statuses = [np.random.randint(0, 3) for _ in range(len(emptyseatev2))]
            self.current_res_demand_EV2[emptyseatev2] = self.demand_price2(EV2_price, behavior_statuses)
            self.current_seat_price_EV2[emptyseatev2] = EV2_price
            waiting_car_ev2_num = EV2_arrival_num - self.current_empnum_seat_EV2
            
            self.waiting_seat_EV2_num = np.sum(self.waiting_seat_EV2)
            if self.waiting_seat_EV2_num >= waiting_car_ev2_num:
                empty_waiting_seat_ev2 = np.where(self.waiting_seat_EV2 == 0)[0]
                self.waiting_seat_EV2[empty_waiting_seat_ev2[:int(waiting_car_ev2_num)]] = 1
                chosen = len(empty_waiting_seat_ev2[:int(waiting_car_ev2_num)])
                behavior_statuses = [np.random.randint(0, 3) for _ in range(chosen)]
                self.waiting_seat_EV2_demand[empty_waiting_seat_ev2[:int(waiting_car_ev2_num)]] = self.demand_price2(EV2_price, behavior_statuses)
                self.waiting_seat_EV2_price[empty_waiting_seat_ev2[:int(waiting_car_ev2_num)]] = EV2_price
            else:
                empty_waiting_seat_ev2 = np.where(self.waiting_seat_EV2 == 0)[0]
                self.waiting_seat_EV2[empty_waiting_seat_ev2] = 1
                behavior_statuses = [np.random.randint(0, 3) for _ in range(len(empty_waiting_seat_ev2))]
                self.waiting_seat_EV2_demand[empty_waiting_seat_ev2] = self.demand_price2(EV2_price, behavior_statuses)
                self.waiting_seat_EV2_price[empty_waiting_seat_ev2] = EV2_price
            

    ### Use Graph Neural Network for Forecasting the marginal prices
    def compute_marginal_price(self, EV1_ch_rate_sum, EV2_ch_rate_sum):
        EV_station_demand = np.array([EV1_ch_rate_sum, EV2_ch_rate_sum])
        self.ppc['bus'][self.busindex, 2] = self.dem_inj[self.busindex] + EV_station_demand/50

        self.result = runopf(self.ppc, self.ppopt)
        Price_margin =self.result['nln']['mu']['u']['Pmis']

        return Price_margin[self.busindex[0]], Price_margin[self.busindex[1]]


    def update_current_reward(self, EV1_ch_rate, EV2_ch_rate):
        EV1_ch_rate_sum = EV1_ch_rate.sum()
        EV2_ch_rate_sum = EV2_ch_rate.sum()
        self.EV1_ch_rate_sum_list.append(EV1_ch_rate_sum)
        self.EV2_ch_rate_sum_list.append(EV2_ch_rate_sum)

        if (self.iter+1)%100 == 0:
            self.EV1_ch_rate_sum_mean = np.array(self.EV1_ch_rate_sum_list).mean()
            self.EV2_ch_rate_sum_mean = np.array(self.EV2_ch_rate_sum_list).mean()
            self.Pr_ev1, self.Pr_ev2 = self.compute_marginal_price(self.EV1_ch_rate_sum_mean, self.EV2_ch_rate_sum_mean)
            self.EV1_ch_rate_sum_list = []
            self.EV2_ch_rate_sum_list = []


        rewardEV1 = np.sum(EV1_ch_rate * self.current_seat_statue_EV1 * self.current_seat_price_EV1) - self.Pr_ev1/2e3 * np.sum(EV1_ch_rate)
        rewardEV2 = np.sum(EV2_ch_rate * self.current_seat_statue_EV2 * self.current_seat_price_EV2) - self.Pr_ev2/2e3 * np.sum(EV2_ch_rate) 

        return rewardEV1, rewardEV2, self.Pr_ev1, self.Pr_ev2, self.EV1_ch_rate_sum_mean, self.EV2_ch_rate_sum_mean

        
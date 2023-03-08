import numpy as np
from Funcs_Logic_DP import get_price
from copy import deepcopy

class DPModel: 
    def __init__(self, N, Start, End, house, merged, battery): 
        self.N = N
        self.Start=Start
        self.End=End
        self.demand = merged.loc[Start:End]['cons_'+house]
        self.prod = merged.loc[Start:End]['prod_'+house]
        self.timeline = merged.loc[Start:End].index
        self.sp = merged.loc[Start:End]["SpotPriceDKK"]/1000
        self.battery = battery
        
        #Compute state space once
        self.s = np.arange(0.0, self.battery.max_capacity+0.1, 0.1)
        for i in range(len(self.s)):
            self.s[i] = int(self.s[i]*10)/10
        
    def f(self, x, u, w, k):
        charge = u[0] 
        
        self.battery.current_capacity = x
        self.battery.charge(charge)
        
        return self.battery.get_current_capacity()
    
    def g(self, x, u, w, k):
        yieldd = self.get_yield(k)
        charge = u[0]
        
        return get_price(yieldd-charge,self.sp[k],0.1)
    
    def gN(self, x):
        return 0.0
    
    def S(self, k):
        return self.s
    
    def A(self, x, k):
        yieldd = self.get_yield(k)
        amount = yieldd
        
        bat_copy=deepcopy(self.battery)
        bat_copy.current_capacity = x
        bat_copy.degrade(1)
        bat_copy.current_capacity = int(bat_copy.current_capacity*10)/10 
        
        x = bat_copy.get_current_capacity()
        
        #Get the discharge to charge range (yield_bat_range)
        if yieldd <=0:
            discharge_amount = amount
            if discharge_amount<-self.battery.max_charge:
                discharge_amount = -self.battery.max_charge 
            
            if discharge_amount + x <0:
                discharge_amount = -x
            
            yield_bat_range = np.round(np.arange(discharge_amount,0+0.01,0.1),2)
        else:
            if amount>self.battery.max_charge:
                amount = self.battery.max_charge
                
            if x+amount>self.battery.max_capacity:
                amount = self.battery.max_capacity-x

            discharge_amount = -x
            if discharge_amount <-self.battery.max_charge:
                discharge_amount = -self.battery.max_charge
            
            yield_bat_range = np.round(np.arange(discharge_amount,amount+0.01,0.1),2)
        
        
        #Calculate if we're allowed to buy charge (if battery can fill max_charge)
        if yield_bat_range[-1]<self.battery.max_charge:
            num_buying_states = self.battery.max_charge*10-int(yield_bat_range[-1]*10)
            num_buy_cap       = self.battery.max_capacity*10-(int(x*10)+int(yield_bat_range[-1]*10))
        
            #But can't charge more that max_capacity
            if num_buying_states>num_buy_cap:
                num_buying_states=num_buy_cap
        else:
            num_buying_states=0
        
        
        actions = np.empty((len(yield_bat_range)+num_buying_states,2))
        
        no_buy = np.array([0 for i in range(len(yield_bat_range))])
        
        #If we can buy
        if num_buying_states>0:
            charge_limit = yield_bat_range[-1] + num_buying_states*0.1
            buy_limit    = num_buying_states*0.1
            
            full_bat_range = np.round(np.arange(discharge_amount,charge_limit+0.01,0.1),2)
            
            buying = np.round(np.arange(0.1,buy_limit+0.01,0.1),2)
            full_buying_range = np.append(no_buy,buying)
        else:
            full_bat_range = yield_bat_range
            full_buying_range = no_buy
        
        actions[:,0] = full_bat_range
        actions[:,1] = full_buying_range
        return actions
    
    def get_yield(self,k):
        return int((self.prod[k] - self.demand[k])*10)/10

if __name__ == "__main__":
    print("This file is meant to be imported")
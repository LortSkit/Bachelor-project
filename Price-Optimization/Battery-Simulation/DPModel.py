import numpy as np
from Funcs_Logic_DP import get_price

class DPModel: 
    def __init__(self, N, Start, End, house, merged, battery_model): 
        self.N = N
        self.demand = merged.loc[Start:End]['cons_'+house]
        self.prod = merged.loc[Start:End]['prod_'+house]
        self.timeline = merged.loc[Start:End].index
        self.sp = merged.loc[Start:End]["SpotPriceDKK"]/1000
        self.battery = battery_model
        
        #Compute state space once
        self.s = np.arange(0.0, self.battery.max_capacity+0.1, 0.1)
        for i in range(len(self.s)):
            self.s[i] = int(self.s[i]*10)/10
        
    def f(self, x, u, w, k):
        charge = u[0] #production and bought
        
        self.battery.current_capacity = x
        self.battery.charge(charge, True)
        
        return self.battery.get_current_capacity()
    
    def g(self, x, u, w, k):
        yieldd = self.prod[k] - self.demand[k]
        charge = np.round(u[0] - u[1],2) #charge from production
        buy = u[1] #bought charge
        
        return get_price(yieldd-charge,self.timeline[k],self.sp[k],0.1)+get_price(-buy,self.timeline[k],self.sp[k],0.1)
    
    def gN(self, x):
        return 0.0
    
    def S(self, k):
        return self.s
    
    def A(self, x, k): #Doesn't account for buying extra
        yieldd = self.prod[k] - self.demand[k]
        amount = yieldd
        
        #Degrade capacity
        reset = self.battery.get_current_capacity() #Needed?
        
        self.battery.current_capacity = x
        self.battery.degrade(1)
        self.battery.current_capacity = int(self.battery.current_capacity*10)/10 
        
        x = self.battery.get_current_capacity()
        
        self.battery.current_capacity = reset       #Needed?
        
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
            
            if discharge_amount + x <0:
                discharge_amount = -x
            
            yield_bat_range = np.round(np.arange(discharge_amount,amount+0.01,0.1),2)
        
        
        #Calculate if we're allowed to buy charge (if battery can fill max_charge)
        num_buying_states = self.battery.max_charge*10-int(yield_bat_range[-1]*10)if yield_bat_range[-1]<7 else 0
        num_buy_cap       = self.battery.max_capacity*10-(int(x*10)+int(yield_bat_range[-1]*10)) if yield_bat_range[-1]<7 else 0
        
        #But can't charge more that max_capacity
        if num_buying_states>num_buy_cap:
            num_buying_states=num_buy_cap
        
        actions = np.round(np.empty((len(yield_bat_range)+num_buying_states,2)),2)
        no_buy = np.round(np.zeros((len(yield_bat_range))),2)
        
        charge_limit = min(np.round(self.battery.max_capacity-x,2),7)
        
        #If we can buy
        if num_buying_states>0:
            
            full_bat_range = np.round(np.arange(discharge_amount,charge_limit+0.01,0.1),2)
            
            buying = np.round(np.arange(0.1,7-yield_bat_range[-1]+0.01,0.1)[:num_buying_states],2)
            full_buying_range = np.round(np.append(no_buy,buying),2)
        else:
            full_bat_range = yield_bat_range
            full_buying_range = no_buy
        

        actions[:,0] = full_bat_range
        actions[:,1] = full_buying_range
        return actions

if __name__ == "__main__":
    print("This is a class meant to be imported")
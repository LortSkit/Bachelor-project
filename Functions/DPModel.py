import numpy as np
import pandas as pd
from Funcs_Logic_DP import get_price, get_emissions, policy_rollout, action_rollout, DP_stochastic
from copy import deepcopy

class DPModel: 
    def __init__(self, Start, End, house, merged, battery,degrade=False,ints=False,acts=None,acts_range=None): 
        self.N = len(pd.date_range(start=Start,end=End,freq="h"))
        self.Start=Start
        self.End=End
        self.demand = merged.loc[Start:End]['cons_'+house]
        self.prod = merged.loc[Start:End]['prod_'+house]
        self.timeline = merged.loc[Start:End].index
        self.sp = merged.loc[Start:End]["SpotPriceDKK"]/1000
        self.battery = battery
        self.degrade = degrade
        self.ints = ints
        self.acts = acts
        self.acts_range = acts_range
        
        #Compute state space once
        if ints:
            self.s = np.arange(0, self.battery.max_capacity+1, 1)
        else:
            self.s = np.round(np.arange(0.0, self.battery.max_capacity+0.01, 0.1),2)
        
    def f(self, x, u, w, k):
        charge = u[0] 
        
        self.battery.current_capacity = x
        self.battery.charge(charge)
        
        if self.ints:
            self.battery.current_capacity = int(self.battery.current_capacity)
        
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
        #Ak(xk) = {− min(mc, xk), . . . , min(mc, cap − xk)}
        
        if self.degrade:
            bat_copy=deepcopy(self.battery)
            bat_copy.current_capacity = x
            bat_copy.degrade(1)
            bat_copy.current_capacity = int(bat_copy.current_capacity*10)/10 

            x = bat_copy.get_current_capacity()
        
        if self.ints:
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),min(self.battery.max_charge,self.battery.max_capacity-x)+1,1),2)
        
        elif (not self.acts_range is None) and (not self.acts is None):
            
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),min(self.battery.max_charge,self.battery.max_capacity-x)+0.01,0.1),2)
            
            below = ranged[ranged<=self.acts[k]][-int((self.acts_range+0.1)*10):] 
            above = ranged[ranged>self.acts[k]][:int(self.acts_range*10)] 
            ranged = np.append(below,above)
        else:
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),min(self.battery.max_charge,self.battery.max_capacity-x)+0.01,0.1),2)
        
        actions = np.empty((len(ranged),1))
        actions[:,0] = ranged
        return actions
    
    def get_yield(self,k):
        return float(f"{self.prod[k] - self.demand[k]:.1f}")
    
class DPModel_c(DPModel):
    def __init__(self, Start, End, house, merged, battery,degrade=False,ints=False,acts=None,acts_range=None): 
        super().__init__(Start, End, house, merged, battery,degrade,ints,acts,acts_range)
        self.ep = merged.loc[Start:End]["CO2Emission"]/1000
    
    def g(self, x, u, w, k):
        yieldd = self.get_yield(k)
        charge = u[0]
        
        return get_emissions(yieldd-charge,self.ep[k]) 

class DPModel_both(DPModel_c):
    def __init__(self, Start, End, house, merged, battery,degrade=False,ints=False,acts=None,acts_range=None,ratio=0.5): 
        super().__init__(Start, End, house, merged, battery,degrade,ints,acts,acts_range)
        self.ratio = ratio
        self.sp = self.norm(self.sp)
        self.ep = self.norm(self.ep)
    
    def norm(self,x):
        return (x-np.min(x))/(np.max(x)-np.min(x))
    
    def g(self, x, u, w, k):
        yieldd = self.get_yield(k)
        charge = u[0]
        
        return (1-self.ratio)*get_price(yieldd-charge,self.sp[k],0.1)+self.ratio*get_emissions(yieldd-charge,self.ep[k]) 
    

def _model_choice(model_name):
    if model_name.lower()[0]=="p":
        return DPModel
    elif model_name.lower()[0]=="c":
        return DPModel_c
    elif model_name.lower()[0]=="b":
        return DPModel_both
    
    raise Exception("Input must be either 'price', 'carbon', or 'both'!!")
    
def _DP(model_name,Start,End,house,merged,DPbat,byday,ints,degrade,verbose,ratio):
    model = _model_choice(model_name)
    N=len(pd.date_range(start=Start,end=End,freq="h"))
    
    DPbat_ints = deepcopy(DPbat)
    
    series_battery_DP = pd.DataFrame(columns=merged.columns)
    
    Start_i = Start
    
    num_loops = int(np.ceil(N/24)) if byday else 1
    remainder = N%24
    length = 24 if byday else N
    for i in range(num_loops):
        if byday and i == num_loops-1:
            length = length if remainder == 0 else remainder
            
        End_i = pd.date_range(start=Start_i,periods=length,freq="h")[-1]
        
        if verbose:
            print(f"Period from {Start_i} to {End_i}")

        if ints:
            if model_name.lower()[0]!="b":
                DP_ints = model(Start_i, End_i, house, merged, deepcopy(DPbat_ints),degrade=degrade,ints=True)
            else:
                DP_ints = model(Start_i, End_i, house, merged, deepcopy(DPbat_ints),degrade=degrade,ints=True,ratio=ratio)
            _, pi_ints = DP_stochastic(DP_ints)
            _, _, actions_ints = policy_rollout(DP_ints,pi=lambda x, k: pi_ints[k][x],x0=int(DPbat_ints.get_current_capacity()))
            charge_i = list(actions_ints["charge"])
            
            if model_name.lower()[0]!="b":
                DP = model(Start_i, End_i, house, merged, deepcopy(DPbat), degrade=degrade, acts=charge_i, acts_range=1.5)
            else:
                DP = model(Start_i, End_i, house, merged, deepcopy(DPbat), degrade=degrade, acts=charge_i, acts_range=1.5, ratio=ratio)

        else:
            if model_name.lower()[0]!="b":
                DP = model(Start_i, End_i, house, merged,deepcopy(DPbat), degrade=degrade)
            else:
                DP = model(Start_i, End_i, house, merged,deepcopy(DPbat), degrade=degrade, ratio=ratio)
        
        _, pi = DP_stochastic(DP)
        _, _, actions = policy_rollout(DP,pi=lambda x, k: pi[k][x],x0=DPbat.get_current_capacity())
        series_battery_DP_i  = action_rollout(merged.loc[Start_i:End_i], DPbat, actions)

        series_battery_DP = pd.concat([series_battery_DP,series_battery_DP_i])
        
        Start_i= pd.date_range(start=End_i,periods=2,freq="h")[-1]

    series_battery_DP["cost_cummulative"] = series_battery_DP["cost"].cumsum(axis=0)
    series_battery_DP["emission_cummulative"] = series_battery_DP["emission"].cumsum(axis=0) 
    
    return series_battery_DP
    
def DP(Start,End,house,merged,DPbat,byday=True,ints=True,degrade=False,verbose=True):
    return _DP("p",Start,End,house,merged,DPbat,byday,ints,degrade,verbose,None)

def DP_carb(Start,End,house,merged,DPbat,byday=True,ints=True,degrade=False,verbose=True):
    return _DP("c",Start,End,house,merged,DPbat,byday,ints,degrade,verbose,None)

def DP_both(Start,End,house,merged,DPbat,byday=True,ints=True,degrade=False,verbose=True,ratio=0.5):
    return _DP("b",Start,End,house,merged,DPbat,byday,ints,degrade,verbose,ratio)
    
if __name__ == "__main__":
    print("This file is meant to be imported")
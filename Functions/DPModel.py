import numpy as np
import pandas as pd
from Funcs_Logic_DP import get_price, get_emissions, policy_rollout, action_rollout, DP_stochastic
from copy import deepcopy

class DPModel: 
    def __init__(self, N, Start, End, house, merged, battery,degrade=False,ints=False,acts=None,acts_range=None): 
        self.N = N
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

def DP(N,Start,End,house,merged,DPbat,byday=True,ints=False,degrade=False):
    DPbat_ints = deepcopy(DPbat)
    num_loops = N//24 if byday else 1
    series_battery_DP = pd.DataFrame(columns=merged.columns)
    length = 24 if byday else N
    for i in range(num_loops):
        merged_i = merged.loc[Start:End].iloc[i*(length):(i+1)*(length)]

        if ints:
            DP_ints = DPModel(length, merged_i.index[0], merged_i.index[-1], house, deepcopy(merged_i), deepcopy(DPbat_ints),degrade,True)
            _, pi_ints = DP_stochastic(DP_ints)
            _, _, actions_ints = policy_rollout(DP_ints,pi=lambda x, k: pi_ints[k][x],x0=int(DPbat_ints.get_current_capacity()))
            charge_i = list(actions_ints["charge"])

            DP = DPModel(length, merged_i.index[0], merged_i.index[-1], house, merged_i, deepcopy(DPbat),degrade,False,charge_i,1.5)

        else:
            DP = DPModel(length,merged_i.index[0], merged_i.index[-1], house, merged_i,deepcopy(DPbat))
        _, pi = DP_stochastic(DP)
        print()
        _, _, actions = policy_rollout(DP,pi=lambda x, k: pi[k][x],x0=DPbat.get_current_capacity())
        series_battery_DP_i  = action_rollout(merged_i, DPbat, actions)

        series_battery_DP = pd.concat([series_battery_DP,series_battery_DP_i])

    series_battery_DP["cost_cummulative"] = series_battery_DP["cost"].cumsum(axis=0)
    series_battery_DP["emission_cummulative"] = series_battery_DP["emission"].cumsum(axis=0) 
    
    return series_battery_DP
    
if __name__ == "__main__":
    print("This file is meant to be imported")
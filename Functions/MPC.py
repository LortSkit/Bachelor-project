import numpy as np
import pandas as pd
from gekko import GEKKO
from Merge import merge
from copy import deepcopy
from Funcs_Logic_DP import action_rollout

class MPCModel:
    def __init__(self, house='h16', sbr_val = 0.1, deg_rate=0.0, num_dec=1,max_charge = 7.0,max_cap = 13.0):
        self.house = house
        self.sbr_val = sbr_val
        self.deg_rate = deg_rate
        self.num_dec = num_dec
        self.max_charge = max_charge
        self.max_cap = max_cap
    
    def return_data(self):
        return merge(self.house)
    
    def norm(self,x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    def MPCopt_price(self, df, start_time='2022-06-19 00:00:00', end_time = '2022-06-19 23:00:00',ini_bat_state=0,verbose=False):
        n = len(pd.date_range(start_time, end_time,freq='H'))

        m = GEKKO()
        charge = m.Array(m.Var, n, lb=-self.max_charge, ub=self.max_charge)
        bat_state = m.Array(m.Var, n+1, lb=0.0, ub=self.max_cap)

        yieldd = df['yield'].loc[start_time:end_time].to_numpy()
        price = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000

        # Define constraints
        m.Equation(bat_state[0] == ini_bat_state)
        m.Equation([bat_state[i+1] == bat_state[i]*(1-self.deg_rate)+charge[i] for i in range(n)])

        # Define objective
        y = m.Array(m.Var, n)
        s = m.Array(m.Var, n)
        p = m.Array(m.Var, n)

        m.Equations([y[i] == yieldd[i] for i in range(n)])
        m.Equations([s[i] == y[i] - charge[i] for i in range(n)])
        m.Equations([p[i] == price[i] for i in range(n)])

        cumm_cost = sum([m.if3(s[i],-1*s[i]*p[i], -self.sbr_val*s[i]*p[i]) for i in range(n)])
        m.Obj(cumm_cost)

        # Solver details
        m.options.IMODE = 3
        m.options.SOLVER = 3
        m.solve(disp=verbose)

        of = pd.DataFrame()
        of['time'] = df.loc[start_time:end_time].index
        of['prod'] = df['prod_'+self.house].loc[start_time:end_time].to_numpy()
        of['cons'] = df['cons_'+self.house].loc[start_time:end_time].to_numpy()
        of['yield'] = [y[i][0] for i in range(n)]
        of['charge'] = [charge[i][0] for i in range(n)]
        of['state'] = [bat_state[i][0] for i in range(n)]
        of['surplus'] = [s[i][0] for i in range(n)]
        of['price'] = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000
        of['costs'] = [-1*p[i][0]*s[i][0] if s[i][0]<0 else -1*p[i][0]*self.sbr_val*s[i][0] for i in range(n)]
        of['cumm_costs'] = of['costs'].cumsum()
        of['em_rate'] = df['CO2Emission'].loc[start_time:end_time].to_numpy()/1000
        of['emission'] = -1*of['surplus']*of['em_rate']
        of['cumm_em'] = of['emission'].cumsum()
        
        of.set_index(of['time'],inplace=True)
        of.drop(['time'],axis=1,inplace=True)

        return of.round(self.num_dec)
    
    def MPCopt_carb(self, df, start_time='2022-06-19 00:00:00', end_time = '2022-06-19 23:00:00',ini_bat_state=0,verbose=False):
        n = len(pd.date_range(start_time, end_time,freq='H'))

        m = GEKKO()
        charge = m.Array(m.Var, n, lb=-self.max_charge, ub=self.max_charge)
        bat_state = m.Array(m.Var, n+1, lb=0.0, ub=self.max_cap)

        yieldd = df['yield'].loc[start_time:end_time].to_numpy()
        carbon = df['CO2Emission'].loc[start_time:end_time].to_numpy()/1000 # kg/kWh

        # Define constraints
        m.Equation(bat_state[0] == ini_bat_state)
        m.Equation([bat_state[i+1] == bat_state[i]*(1-self.deg_rate)+charge[i] for i in range(n)])

        # Define objective
        y = m.Array(m.Var, n)
        s = m.Array(m.Var, n)
        c = m.Array(m.Var, n)

        m.Equations([y[i] == yieldd[i] for i in range(n)])
        m.Equations([s[i] == y[i] - charge[i] for i in range(n)])
        m.Equations([c[i] == carbon[i] for i in range(n)])

        cumm_cost = sum([-1*s[i]*c[i] for i in range(n)])
        m.Obj(cumm_cost)

        # Solver details
        m.options.IMODE = 3
        m.options.SOLVER = 3
        m.solve(disp=verbose)

        of = pd.DataFrame()
        of['time'] = df.loc[start_time:end_time].index
        of['prod'] = df['prod_'+self.house].loc[start_time:end_time].to_numpy()
        of['cons'] = df['cons_'+self.house].loc[start_time:end_time].to_numpy()
        of['yield'] = [y[i][0] for i in range(n)]
        of['charge'] = [charge[i][0] for i in range(n)]
        of['state'] = [bat_state[i][0] for i in range(n)]
        of['surplus'] = [s[i][0] for i in range(n)]
        of['price'] = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000
        of['costs'] = [-1*of['price'].loc[i]* of['surplus'].loc[i] if of['surplus'].loc[i]<0 else -1*self.sbr_val*of['price'].iloc[i]* of['surplus'].iloc[i] for i in range(n)]
        of['cumm_costs'] = of['costs'].cumsum()
        of['em_rate'] = [carbon[i] for i in range(n)]
        of['emission'] = -1*of['surplus']*of['em_rate']
        of['cumm_em'] = of['emission'].cumsum()

        of.set_index(of['time'],inplace=True)
        of.drop(['time'],axis=1,inplace=True)
        
        return of.round(self.num_dec)
    
    def MPCopt_(self, df, start_time='2022-06-19 00:00:00', end_time = '2022-06-19 23:00:00',ini_bat_state=0,verbose=False,ratio=0.5):
        n = len(pd.date_range(start_time, end_time,freq='H'))

        m = GEKKO()
        charge = m.Array(m.Var, n, lb=-self.max_charge, ub=self.max_charge)
        bat_state = m.Array(m.Var, n+1, lb=0.0, ub=self.max_cap)

        yieldd = df['yield'].loc[start_time:end_time].to_numpy()
        
        carbon_ = df['CO2Emission'].loc[start_time:end_time].to_numpy()/1000 # kg/kWh
        price_ = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000 # DKK/kWh
        carbon = self.norm(carbon_)
        price = self.norm(price_)
        
        # Define constraints
        m.Equation(bat_state[0] == ini_bat_state)
        m.Equation([bat_state[i+1] == bat_state[i]*(1-self.deg_rate)+charge[i] for i in range(n)])

        # Define objective
        y = m.Array(m.Var, n)
        s = m.Array(m.Var, n)
        c = m.Array(m.Var, n)
        p = m.Array(m.Var, n)

        m.Equations([y[i] == yieldd[i] for i in range(n)])
        m.Equations([s[i] == y[i] - charge[i] for i in range(n)])
        m.Equations([c[i] == carbon[i] for i in range(n)])
        m.Equations([p[i] == price[i] for i in range(n)])
        
        c_cost = sum([-1*s[i]*c[i] for i in range(n)])
        p_cost = sum([m.if3(s[i],-1*s[i]*p[i], -self.sbr_val*s[i]*p[i]) for i in range(n)])
        m.Obj((1-ratio)*p_cost+ratio*c_cost)
        
        # Solver details
        m.options.IMODE = 3
        m.options.SOLVER = 3
        m.solve(disp=verbose)
        
        #print([p[i] for i in range(n)])
        #print([c[i] for i in range(n)])
        
        of = pd.DataFrame()
        of['time'] = df.loc[start_time:end_time].index
        of['prod'] = df['prod_'+self.house].loc[start_time:end_time].to_numpy()
        of['cons'] = df['cons_'+self.house].loc[start_time:end_time].to_numpy()
        of['yield'] = [y[i][0] for i in range(n)]
        of['charge'] = [charge[i][0] for i in range(n)]
        of['state'] = [bat_state[i][0] for i in range(n)]
        of['surplus'] = [s[i][0] for i in range(n)]
        of['price'] = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000
        of['costs'] = [-1*of['price'].loc[i]* of['surplus'].loc[i] if of['surplus'].loc[i]<0 else -1*self.sbr_val*of['price'].iloc[i]* of['surplus'].iloc[i] for i in range(n)]
        of['cumm_costs'] = of['costs'].cumsum()
        of['em_rate'] = df['CO2Emission'].loc[start_time:end_time].to_numpy()/1000
        of['emission'] = -1*of['surplus']*of['em_rate']
        of['cumm_em'] = of['emission'].cumsum()

        of.set_index(of['time'],inplace=True)
        of.drop(['time'],axis=1,inplace=True)
        
        return of.round(self.num_dec)
    
def MPC(Start,End,merged,MPCbat,MPCModel,byday=True):
    N=len(pd.date_range(start=Start,end=End,freq="h"))
    series_battery_MPC_price=pd.DataFrame()
    series_battery_MPC_carb=pd.DataFrame()
    
    MPCbat_price = deepcopy(MPCbat)
    MPCbat_carb = deepcopy(MPCbat)

    num_loops = int(np.ceil(N/24)) if byday else 1
    remainder = N%24
    length = 24 if byday else N
    Start_i=Start
    for i in range(num_loops):
        if byday and i == num_loops-1:
            length = length if remainder == 0 else remainder

        End_i = pd.date_range(start=Start_i,periods=length,freq="h")[-1]

        print(f"Period from {Start_i} to {End_i}")

        actions_price = MPCModel.MPCopt_price(merged, Start_i, End_i, MPCbat_price.get_current_capacity())

        series_battery_MPC_price_i = action_rollout(merged.loc[Start_i:End_i], MPCbat_price, actions_price)


        actions_carb = MPCModel.MPCopt_carb(merged, Start_i, End_i, MPCbat_carb.get_current_capacity())

        series_battery_MPC_carb_i = action_rollout(merged.loc[Start_i:End_i], MPCbat_carb, actions_carb)


        series_battery_MPC_price = pd.concat([series_battery_MPC_price,series_battery_MPC_price_i])
        series_battery_MPC_carb  = pd.concat([series_battery_MPC_carb,series_battery_MPC_carb_i])

        Start_i= pd.date_range(start=End_i,periods=2,freq="h")[-1]


    series_battery_MPC_price["cost_cummulative"] = series_battery_MPC_price["cost"].cumsum(axis=0)
    series_battery_MPC_price["emission_cummulative"] = series_battery_MPC_price["emission"].cumsum(axis=0)    

    series_battery_MPC_carb["cost_cummulative"] = series_battery_MPC_carb["cost"].cumsum(axis=0)
    series_battery_MPC_carb["emission_cummulative"] = series_battery_MPC_carb["emission"].cumsum(axis=0)  
    
    return series_battery_MPC_price,series_battery_MPC_carb
    
if __name__ == "__main__":
    print("This file is meant to be imported")
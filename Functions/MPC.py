import numpy as np
import pandas as pd
from gekko import GEKKO
from Merge import merge
from copy import deepcopy
from Logic import action_rollout

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
        m.options.MAX_ITER = 10000
        m.options.IMODE = 3
        solvers = [3,1,2] 
        for solver in solvers:
            try:
                if verbose:
                    print(f'Trying solver {solver}')
                m.options.SOLVER = solver
                m.solve(disp=verbose)
                break 
            except Exception as e:
                if verbose:
                    print(f"Solver {solver} failed with error: {e}")
        else:
            print("All solvers failed")

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

        #Proper rounding of charge to one decimal
        values = of["charge"].to_numpy()
        for i in range(len(values)):
            values[i] = f"{values[i]:.1f}"

        of["charge"] = values
        
        return of
    
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
        m.options.MAX_ITER = 10000
        m.options.IMODE = 3
        solvers = [3,1,2] 
        for solver in solvers:
            try:
                if verbose:
                    print(f'Trying solver {solver}')
                m.options.SOLVER = solver
                m.solve(disp=verbose)
                break 
            except Exception as e:
                if verbose:
                    print(f"Solver {solver} failed with error: {e}")
        else:
            print("All solvers failed")

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
        
        #Proper rounding of charge to one decimal
        values = of["charge"].to_numpy()
        for i in range(len(values)):
            values[i] = f"{values[i]:.1f}"

        of["charge"] = values
        
        return of
    
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
        m.options.MAX_ITER = 10000
        m.options.IMODE = 3
        solvers = [3,1,2] 
        for solver in solvers:
            try:
                if verbose:
                    print(f'Trying solver {solver}')
                m.options.SOLVER = solver
                m.solve(disp=verbose)
                break 
            except Exception as e:
                if verbose:
                    print(f"Solver {solver} failed with error: {e}")
        else:
            print("All solvers failed")
        
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
        
        #Proper rounding of charge to one decimal
        values = of["charge"].to_numpy()
        for i in range(len(values)):
            values[i] = f"{values[i]:.1f}"

        of["charge"] = values
        
        return of
    
def _model_choice(model_name,MPCModel):
    if model_name.lower()[0]=="p":
        return MPCModel.MPCopt_price
    elif model_name.lower()[0]=="c":
        return MPCModel.MPCopt_carb
    elif model_name.lower()[0]=="b":
        return MPCModel.MPCopt_
    
    raise Exception("Input must be either 'price', 'carbon', or 'both'!!")
    
def _MPC(model_name,Start,End,merged,MPCbat,MPCModel,byday,verbose,ratio):
    model = _model_choice(model_name,MPCModel)
    N=len(pd.date_range(start=Start,end=End,freq="h"))
    series_battery_MPC=pd.DataFrame()

    num_loops = int(np.ceil(N/24)) if byday else 1
    remainder = N%24
    length = 24 if byday else N
    Start_i=Start
    for i in range(num_loops):
        if byday and i == num_loops-1:
            length = length if remainder == 0 else remainder

        End_i = pd.date_range(start=Start_i,periods=length,freq="h")[-1]
        
        if verbose:
            print(f"Period from {Start_i} to {End_i}")

        if model_name.lower()[0]!="b":
            actions = model(merged, Start_i, End_i, MPCbat.get_current_capacity(),verbose)
        else:
            actions = model(merged, Start_i, End_i, MPCbat.get_current_capacity(),verbose,ratio)

        series_battery_MPC_i = action_rollout(merged.loc[Start_i:End_i], MPCbat, actions)

        
        series_battery_MPC  = pd.concat([series_battery_MPC,series_battery_MPC_i])

        Start_i= pd.date_range(start=End_i,periods=2,freq="h")[-1]


    series_battery_MPC["cost_cummulative"] = series_battery_MPC["cost"].cumsum(axis=0)
    series_battery_MPC["emission_cummulative"] = series_battery_MPC["emission"].cumsum(axis=0)    
    
    return series_battery_MPC

def MPC(Start,End,merged,MPCbat,MPCModel,byday=True,verbose=True):
    return _MPC("p",Start,End,merged,MPCbat,MPCModel,byday,verbose,None)
    
def MPC_carb(Start,End,merged,MPCbat,MPCModel,byday=True,verbose=True):
    return _MPC("c",Start,End,merged,MPCbat,MPCModel,byday,verbose,None)

def MPC_both(Start,End,merged,MPCbat,MPCModel,ratio=0.5,byday=True,verbose=True):
    return _MPC("b",Start,End,merged,MPCbat,MPCModel,byday,verbose,ratio)
    
if __name__ == "__main__":
    print("This file is meant to be imported")
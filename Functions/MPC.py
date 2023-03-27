import numpy as np
import pandas as pd
from gekko import GEKKO
from Merge import merge

class MPC:
    def __init__(self, house='h16', sbr_val = 0.1, deg_rate=0.0, num_dec=1,max_charge = 7.0,max_cap = 13.0):
        self.house = house
        self.sbr_val = sbr_val
        self.deg_rate = deg_rate
        self.num_dec = num_dec
        self.max_charge = max_charge
        self.max_cap = max_cap

    def MPCopt(self, df, start_time='2022-06-19 00:00:00', end_time = '2022-06-19 23:00:00',ini_bat_state=0):
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
        m.solve()

        of = pd.DataFrame()
        #of['time'] = df['time'].loc[start_time:end_time].to_numpy()
        #of['prod'] = df['prod'].loc[start_time:end_time].to_numpy()
        #of['cons'] = df['cons'].loc[start_time:end_time].to_numpy()
        #of['yield'] = [y[i][0] for i in range(n)]
        of['charge'] = [charge[i][0] for i in range(n)]
        #of['state'] = [bat_state[i][0] for i in range(n)]
        #of['surplus'] = [s[i][0] for i in range(n)]
        #of['price'] = df['SpotPriceDKK'].loc[start_time:end_time].to_numpy()/1000
        #of['costs'] = [-1*p[i][0]*s[i][0] if s[i][0]<0 else -1*p[i][0]*self.sbr_val*s[i][0] for i in range(n)]
        #of['cumm_costs'] = of['costs'].cumsum()

        return of['charge'].round(self.num_dec)
        
    def return_data(self):
        return merge(self.house)

if __name__ == "__main__":
    print("This file is meant to be imported")
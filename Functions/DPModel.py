import numpy as np
import pandas as pd
from Battery import Battery
from Logic import get_price, get_emissions, action_rollout, logic_series_print
from copy import deepcopy
from P2P_Dynamics import EnergyMarket

from itertools import permutations
from IPython.display import display, HTML

class DPModel: 
    '''
    DPModel, which is made for price optimization using a simulated battery
    
    This model should define the problem as:
    State spaces, where states are labled "x" (battery capacity)
    Action space, where actions are labled "u" (charge action)
    Transition function, called "f" (from state to new state via action)
    Cost function, called "g" (function to optimize, lower is better)
    
    Usage: For input in the DP_stochastic and policy_rollout functions
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
    
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    degrade: bool, Flag that tells model whether battery should degrade when charging battery
            
             This is by default False, meaning no degrading is done
             
    ints: bool, Flag that tells model whether to look at a smaller problem
            
          Problem is made smaller by, instead of have a discretization stepsize of 0.1,
          having a discretization stepsize of 1 for both the state and actions space
          
          This speeds the problem up by a lot, slightly less optimal 
          
    acts: Pandas series or None, Limits actions space to be close to these actions
            
          Usually after having run a model with the ints flag set to True using DP_stochastic,
          and then running a policy rollout, then running the model a second time with the 
          ints flag set to False and inputting the action column into acts. Doing this will
          result in returning an action space limited by acts and acts_range, where for each
          timestep k, the usual range of the action space is limited by taking the acts_range
          values below and above acts[k]
          
          If None, the action space is not limited
          
    acts_range: float, Limits the action space using this value
            
                (See acts explanation)
                By default this value is None, but for the report usually 1.5 is used
          
    
    Example: THIS IS AN INTERNAL CLASS, SHOULD NOT BE IMPORTED
    
    
    Additional attributes:
    
    N: int, Length of period calculated using Start and End
            
       Used as input by DP_stochastic 
       
    yieldd: Pandas series, The yield either from predictions or raw data
            
           yield = production - consumption  
          
    timeline: Pandas DatetimeIndex, Is simply the indices obtained from merged
            
              Is an iterable of all indices between Start end End
              
    sp: Pandas series, The spotprices obtained from merged
            
        Is used in the cost function, g
      
    s: float numpy array if ints=False otherwise int numpy array, State space
       
       Since the state space doesn't change for any timesteps, it's calculated
       during initilization and stored to this attribute
       
       If discretized to integers when ints=True, this is and integer numpy
       arry, and if ints=False it's a float numpy array
                                
    Functions (check their docs):
    
    f: float * float * bool * int -> float
    
    g: float * float * bool * int -> float
    
    gN: float -> float
    
    S: int -> float numpy array
    
    A: float * int -> float numpy array
    
    get_yield: int -> float
    '''
    def __init__(self, Start, End, merged, battery,degrade=False,ints=False,acts=None,acts_range=None): 
        self.N = len(pd.date_range(start=Start,end=End,freq="h"))
        self.Start=Start
        self.End=End
        self.yieldd = merged.loc[Start:End]["yield"]
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
        '''
        Charges/discharges u kWh to/from battery with capacity x at timestep k

        Return type: float

        Usage: Used by DP_stochastic and Policy rollout to transition battery state


        Input:

        x: float, capacity of battery; state
        
           This is the capacity in the battery, u is how much is charged/discharged
           
        u: float, how much is charged; action
        
           This is how much is charged/discharge to/from the battery, x is the capacity
           
        w: bool, Is unused
        
           Present only to adhere to a general DPModel standard, where
           w would represent whether problem behaves stochastically.
           If False, deterministic, stochastic otherwise

        k: int, timestep
           
           There are N+1 timesteps, 0-indexed


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        charge = u[0] 
        
        self.battery.current_capacity = x
        self.battery.charge(charge, degrade=self.degrade)
        
        if self.ints:
            self.battery.current_capacity = int(self.battery.current_capacity)
        
        return self.battery.get_current_capacity()
    
    def g(self, x, u, w, k):
        '''
        Calculates cost of charging/discharging u kWh to/from battery with capacity x at timestep k

        Return type: float

        Usage: Used by DP_stochastic and Policy rollout to calculate cost


        Input:

        x: float, capacity of battery; state
        
           This is the capacity in the battery, u is how much is charged/discharged
           
        u: float, how much is charged; action
        
           This is how much is charged/discharge to/from the battery, x is the capacity
           
        w: bool, Is unused
        
           Present only to adhere to a general DPModel standard, where
           w would represent whether problem behaves stochastically.
           If False, deterministic, stochastic otherwise

        k: int, timestep
           
           There are N+1 timesteps, 0-indexed


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        yieldd = self.get_yield(k)
        charge = u[0]
        
        fee = 1 #transmission fee
        
        return get_price(yieldd-charge,fee+self.sp[k],0.1)
    
    def gN(self, x):
        """
        Cost of reaching end of path
        
        Return type: float
        
        Usage: Used by DP_stochastic and Policy rollout to get cost of reaching end of path
        
        Input:
        
        x: float, capacity of battery; state
        
           Ignored here as the cost of reaching end state is nothing.
           Is here to adhere to a general DPModel standard, where
           there may be a cost of being in a given state x by the
           end of the path. Usually there's a cost if x is not a goal
           state, and/or a reward if x is a goal state.
           
           
        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        """
        return 0.0
    
    def S(self, k):
        '''
        Returns the statespace, calculated at initilization
        Mathimatical definition:
        S_k = {0.0, ..., max_capacity}

        Return type: float numpy array or int numpy array

        Usage: Used by DP_stochastic and Policy rollout to get available states


        Input:

        k: int, timestep
           
           Ignored here as this problems state space is the same at all
           timesteps. Is here to adhere to a general DPModel standard,
           where some states are only available at certain timesteps.


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        return self.s
    
    def A(self, x, k):
        '''
        Returns the action space available at state x at timestep k
        Mathimatical definition:
        #A_k(x_k) = {−min(max_capacity, x_k), ..., min(max_capacity, max_capacity − x_k)}

        Return type: float numpy array

        Usage: Used by DP_stochastic and Policy rollout to get available actions


        Input:

        x: float, capacity of battery; state
        
           This is the capacity in the battery

        k: int, timestep
           
           Ignored here as this problems state space is the same at all
           timesteps. Is here to adhere to a general DPModel standard,
           where some states are only available at certain timesteps.


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        
        if self.degrade:
            bat_copy=deepcopy(self.battery)
            bat_copy.current_capacity = x
            bat_copy.degrade(1)
            bat_copy.current_capacity = bat_copy.round_one_decimal(bat_copy.current_capacity)

            x = bat_copy.get_current_capacity()
        
        if self.ints:
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),min(self.battery.max_charge,self.battery.max_capacity-x)+1,1),2)
        
        elif (not self.acts_range is None) and (not self.acts is None):
            
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),
                                         min(self.battery.max_charge,self.battery.max_capacity-x)+0.01,0.1),2)
            
            below = ranged[ranged<=self.acts[k]][-int((self.acts_range+0.1)*10):] 
            above = ranged[ranged>self.acts[k]][:int(self.acts_range*10)] 
            ranged = np.append(below,above)
        else:
            ranged = np.round(np.arange(-min(self.battery.max_charge,x),
                                         min(self.battery.max_charge,self.battery.max_capacity-x)+0.01,0.1),2)
        
        actions = np.empty((len(ranged),1))
        actions[:,0] = ranged
        return actions
    
    def get_yield(self,k):
        """
        Returns class attribute "yieldd" (the yield)
        
        Return type: float
        """
        return self.yieldd[k]
    
class DPModel_c(DPModel):
    '''
    DPModel, which is made for carbon emissions optimization using a simulated battery
    
    This model should define the problem as:
    State spaces, where states are labled "x" (battery capacity)
    Action space, where actions are labled "u" (charge action)
    Transition function, called "f" (from state to new state via action)
    Cost function, called "g" (function to optimize, lower is better)
    
    Usage: For input in the DP_stochastic and policy_rollout functions
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
    
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    degrade: bool, Flag that tells model whether battery should degrade when charging battery
            
             This is by default False, meaning no degrading is done
             
    ints: bool, Flag that tells model whether to look at a smaller problem
            
          Problem is made smaller by, instead of have a discretization stepsize of 0.1,
          having a discretization stepsize of 1 for both the state and actions space
          
          This speeds the problem up by a lot, slightly less optimal 
          
    acts: Pandas series or None, Limits actions space to be close to these actions
            
          Usually after having run a model with the ints flag set to True using DP_stochastic,
          and then running a policy rollout, then running the model a second time with the 
          ints flag set to False and inputting the action column into acts. Doing this will
          result in returning an action space limited by acts and acts_range, where for each
          timestep k, the usual range of the action space is limited by taking the acts_range
          values below and above acts[k]
          
          If None, the action space is not limited
          
    acts_range: float, Limits the action space using this value
            
                (See acts explanation)
                By default this value is None, but for the report usually 1.5 is used
          
    
    Example: THIS IS AN INTERNAL CLASS, SHOULD NOT BE IMPORTED
    
    
    Additional attributes:
    
    N: int, Length of period calculated using Start and End
            
       Used as input by DP_stochastic 
       
    yieldd: Pandas series, The yield either from predictions or raw data
            
           yield = production - consumption  
          
    timeline: Pandas DatetimeIndex, Is simply the indices obtained from merged
            
              Is an iterable of all indices between Start end End
              
    sp: Pandas series, The spotprices obtained from merged
            
        Inherited from DPModel class, unused
        
    ep: Pandas series, The emissions costs obtained from merged
            
        Is used in the cost function, g
      
    s: float numpy array if ints=False otherwise int numpy array, State space
       
       Since the state space doesn't change for any timesteps, it's calculated
       during initilization and stored to this attribute
       
       If discretized to integers when ints=True, this is and integer numpy
       arry, and if ints=False it's a float numpy array
                                
    Functions (check their docs):
    
    f: float * float * bool * int -> float
    
    g: float * float * bool * int -> float
    
    gN: float -> float
    
    S: int -> float numpy array
    
    A: float * int -> float numpy array
    
    get_yield: int -> float
    '''
    def __init__(self, Start, End, merged, battery,degrade=False,ints=False,acts=None,acts_range=None): 
        super().__init__(Start, End, merged, battery,degrade,ints,acts,acts_range)
        self.ep = merged.loc[Start:End]["CO2Emission"]/1000
    
    def g(self, x, u, w, k):
        '''
        Calculates emission cost of charging/discharging u kWh to/from battery with capacity x at timestep k

        Return type: float

        Usage: Used by DP_stochastic and Policy rollout to calculate cost


        Input:

        x: float, capacity of battery; state
        
           This is the capacity in the battery, u is how much is charged/discharged
           
        u: float, how much is charged; action
        
           This is how much is charged/discharge to/from the battery, x is the capacity
           
        w: bool, Is unused
        
           Present only to adhere to a general DPModel standard, where
           w would represent whether problem behaves stochastically.
           If False, deterministic, stochastic otherwise

        k: int, timestep
           
           There are N+1 timesteps, 0-indexed


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        yieldd = self.get_yield(k)
        charge = u[0]
        
        return get_emissions(yieldd-charge,self.ep[k]) 

class DPModel_both(DPModel_c):
    '''
    DPModel, which is made for carbon emissions optimization using a simulated battery
    
    This model should define the problem as:
    State spaces, where states are labled "x" (battery capacity)
    Action space, where actions are labled "u" (charge action)
    Transition function, called "f" (from state to new state via action)
    Cost function, called "g" (function to optimize, lower is better)
    
    Usage: For input in the DP_stochastic and policy_rollout functions
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
    
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    degrade: bool, Flag that tells model whether battery should degrade when charging battery
            
             This is by default False, meaning no degrading is done
             
    ints: bool, Flag that tells model whether to look at a smaller problem
            
          Problem is made smaller by, instead of have a discretization stepsize of 0.1,
          having a discretization stepsize of 1 for both the state and actions space
          
          This speeds the problem up by a lot, slightly less optimal 
          
    acts: Pandas series or None, Limits actions space to be close to these actions
            
          Usually after having run a model with the ints flag set to True using DP_stochastic,
          and then running a policy rollout, then running the model a second time with the 
          ints flag set to False and inputting the action column into acts. Doing this will
          result in returning an action space limited by acts and acts_range, where for each
          timestep k, the usual range of the action space is limited by taking the acts_range
          values below and above acts[k]
          
          If None, the action space is not limited
          
    acts_range: float, Limits the action space using this value
            
                (See acts explanation)
                By default this value is None, but for the report usually 1.5 is used
    
    ratio: float, How much to weight emissions compared to spotprices
    
           Should be between 0 and 1
           
           Used the following way:
           (1-ratio)*spotprices + ratio*emissions
           
    
    Example: THIS IS AN INTERNAL CLASS, SHOULD NOT BE IMPORTED
    
    
    Additional attributes:
    
    N: int, Length of period calculated using Start and End
            
       Used as input by DP_stochastic 
       
    yieldd: Pandas series, The yield either from predictions or raw data
            
           yield = production - consumption  
          
    timeline: Pandas DatetimeIndex, Is simply the indices obtained from merged
            
              Is an iterable of all indices between Start end End
              
    sp: Pandas series, The spotprices obtained from merged
            
        Is used in the cost function, g. Normalized to be equally weighted with
        emissions
        
    ep: Pandas series, The emissions costs obtained from merged
            
        Is used in the cost function, g. Normalized to be equally weighted with
        spotprices
      
    s: float numpy array if ints=False otherwise int numpy array, State space
       
       Since the state space doesn't change for any timesteps, it's calculated
       during initilization and stored to this attribute
       
       If discretized to integers when ints=True, this is and integer numpy
       arry, and if ints=False it's a float numpy array
                                
    Functions (check their docs):
    
    f: float * float * bool * int -> float
    
    g: float * float * bool * int -> float
    
    gN: float -> float
    
    S: int -> float numpy array
    
    A: float * int -> float numpy array
    
    get_yield: int -> float
    
    norm: float iterable -> float iterable
    '''
    def __init__(self, Start, End, merged, battery,degrade=False,ints=False,acts=None,acts_range=None,ratio=0.5): 
        super().__init__(Start, End, merged, battery,degrade,ints,acts,acts_range)
        self.ratio = ratio
        
        fee = 1 #transmission fee
        self.sp = self.sp+fee
        
        self.sp = self.norm(self.sp)
        self.ep = self.norm(self.ep)
    
    def norm(self, arr):
        """
        Normalizes input array
        
        Return type: float iterable, same as input
        
        Usage: normalizes spotprices and emissions so that they're weighted only
               by ratio
               
               
        Input:
        
        ar: float iterable, usually Pandas series
        
            This function is only really used for normalizing the spotsprices
            and emissions so that they're weighted only by the ratio.
            
        
        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        """
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    
    def g(self, x, u, w, k):
        '''
        Calculates weighted cost of both price and emissions of charging/discharging 
        u kWh to/from battery with capacity x at timestep k

        Return type: float

        Usage: Used by DP_stochastic and Policy rollout to calculate cost


        Input:

        x: float, capacity of battery; state
        
           This is the capacity in the battery, u is how much is charged/discharged
           
        u: float, how much is charged; action
        
           This is how much is charged/discharge to/from the battery, x is the capacity
           
        w: bool, Is unused
        
           Present only to adhere to a general DPModel standard, where
           w would represent whether problem behaves stochastically.
           If False, deterministic, stochastic otherwise

        k: int, timestep
           
           There are N+1 timesteps, 0-indexed


        Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE USED OUTSIDE THIS FILE
        '''
        yieldd = self.get_yield(k)
        charge = u[0]
        
        return (1-self.ratio)*get_price(yieldd-charge,self.sp[k],0.1)+self.ratio*get_emissions(yieldd-charge,self.ep[k]) 
    
class DP_central(DPModel):
    def __init__(self, Start, End, merges, houses, battery,degrade=False,ints=False,acts=None,acts_range=None,
                 furthers=None,traj=None, traj_range=None,max_number_states=200):
        #Set class attributes
        super().__init__(Start, End, merges[0], battery,degrade,ints,acts,acts_range)
        self.merges = merges
        self.houses = houses
        self.furthers = furthers
        self.traj = traj
        self.traj_range = traj_range
        self.max_number_states = max_number_states
        
        temp = pd.DataFrame(columns=houses)
        for i in range(len(houses)):
            temp[houses[i]] = merges[i].loc[Start:End]["yield"]

        self.yields = temp
    
        #Compute state space once
        if ints:
            states = np.arange(0, battery.max_capacity+1, 1)
        else:
            states = np.round(np.arange(0.0, battery.max_capacity+0.01, 0.1),2)
        
        self.states = [states for _ in range(len(houses))]
        
        if self.ints and self.furthers is not None:
            for i in range(len(houses)):
                temp = self.states[i]
                self.states[i] = temp[temp%self.furthers[i][0]==self.furthers[i][1]]

        self.s = np.array(np.meshgrid(*self.states)).T.reshape(-1,3)
        self.s = self.s.tolist()
        self.s = [tuple(x) for x in self.s]
    
    def f(self, x, u, w, k):
        ogbat = self.battery
        self.battery = deepcopy(ogbat)
        
        res = []
        for i in range(len(self.houses)):
            self.battery.current_capacity = x[i]
            self.battery.charge(u[i], degrade=self.degrade)

            if self.ints:
                self.battery.current_capacity = int(self.battery.current_capacity)
            
            res.append(self.battery.get_current_capacity())
            
        self.battery = ogbat
        
        return tuple(res)
    
    def g(self, x, u, w, k):
        yields = self.get_yield(k)
        surpluses = [yields[i]-u[i] for i in range(len(u))]
        
        fee = 1 #transmission fee
        
        participants = {self.houses[i]: surpluses[i] for i in range(len(u))}
    
        em = EnergyMarket(participants, self.sp[k], self.sp[k]+fee)
        
        dic = em.get_total_costs()
        
        return sum([dic[house] for house in self.houses])
    
    def S(self, k):
        if (self.traj is not None) and (self.traj_range is not None):
            for i in range(len(self.houses)):
                if self.traj_range[i]==0.0:
                    self.states[i] = self.traj[k][i]
                
            temp = np.array(np.meshgrid(*self.states)).T.reshape(-1,3)
            temp = temp.tolist()
            temp = [tuple(x) for x in temp]
            
            return temp
        
        return self.s
    
    def A(self, x, k):
        states=[]
        acts = self.acts
        self.acts=None
        for i in range(len(x)):
            temp = super().A(x[i], k)
            states.append(temp)
            
        self.acts=acts
        
        if self.ints:
            states = [states[i][np.array(states[i],dtype=int)==states[i]] for i in range(len(x))]
            
        if self.acts is not None and self.acts_range is not None:
            for i in range(len(x)):
                if self.acts_range[i]!=0.0:
                    below =states[i][states[i]<=self.acts[k][i]][-int((self.acts_range[i]+0.1)*10):]
                    above =states[i][states[i]>self.acts[k][i]][:int(self.acts_range[i]*10)]

                    states[i] = np.append(below,above)
                else:
                    states[i] = self.acts[k][i]
        
        if self.ints and self.furthers is not None:
            states = [states[i][(states[i]+x[i])%self.furthers[i][0]==self.furthers[i][1]] for i in range(len(x))]
            
        actions = np.array(np.meshgrid(*states)).T.reshape(-1,3)
        if len(actions)>self.max_number_states:
            idx = np.round(np.linspace(0, len(actions) - 1, self.max_number_states)).astype(int)
            actions = actions[idx]

        actions = list(map(tuple,actions))
        return actions
    
    def get_yield(self,k):
        return self.yields.iloc[k].to_numpy()
    
def DP_stochastic(model):
    """
    This function implements Algorithm 1, The dynamical programming (DP) algorithm, 
    from "Sequential Decision-Making" (Feb 2, 2022) by Tue Herlau
    
    Return type:             dict list * dict list
                 (float -> float) list * (float -> float) list
    
    Usage: This function sees use internally in "DP" function from "DPModel.py",
           and shouldn't be used otherwise
    
    
    Input:
    
    model: DPModel (or DPModel_c or DPModel_both) instance, from "DPModel.py"
    
           This model should define the problem as:
           State spaces, where states are labled "x" (battery capacity)
           Action space, where actions are labled "u" (charge action)
           Transition function, called "f" (from state to new state via action)
           Cost function, called "g" (function to optimize, lower is better)
             
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        for x in model.S(k): 
            
            #w is unused
            w = False

            Qu = {tuple(u): (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for u in model.A(x, k)} 
            umin = min(Qu, key=Qu.get)
            J[k][x] = Qu[umin]
            pi[k][x] = umin

    return J, pi

def policy_rollout(model, pi, x0):
    """
    Rolls out the policy obtained from running model in DP_stochastic, with output
    policy "pi" starting in state x0. The returned "actions" is a dataframe ready
    for action rollout.
    
    Return type: float * float list * Pandas dataframe
    
    Usage: This function sees use internally in "DP" function from "DPModel.py",
           and shouldn't be used otherwise
    
    
    Input:
    
    model: DPModel (or DPModel_c or DPModel_both) instance, from "DPModel.py"
    
           This model should define the problem as:
           State spaces, where states are labled "x" (battery capacity)
           Action space, where actions are labled "u" (charge action)
           Transition function, called "f" (from state to new state via action)
           Cost function, called "g" (function to optimize, lower is better)
           
    pi: (float * int) -> float, a "function" version of the "pi" output from DP_stochastic
    
        
        Should be passed as pi=lambda x, k: pi[k][x], where the list "pi" is from 
        DP_stochastic
        
    x0: float, initial state of battery
        
        Batteries are usually in initial state x0=0, unless this is used
        as a continuation of a previous rollout like, e.g., single house optimization 
    
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    cost = 0
    J, x, trajectory, actions = 0, x0, [x0], []
    for k in range(model.N):
        u = pi(x, k)
        J += model.g(x, u , True, k)
        x = model.f(x, u, True, k)
        trajectory.append(x) # update the list of the trajectory
        actions.append(u) # update the list of the actions
    
    J += model.gN(x)
    actions = pd.DataFrame(actions,columns=['charge'])
    actions.index = pd.date_range(start=model.Start, end=model.End, freq="h")
    return J, trajectory, actions

def p2p_rollout(model, pi, x0):
    """
    Similar to policy rollout, but for p2p
    """
    cost = 0
    surpluses = []
    J, x, trajectory, actions = 0, x0, [x0], []
    for k in range(model.N):
        u = pi(x, k)
        price = model.g(x, u , True, k)
        surpluses.append([model.get_yield(k)[i]-u[i] for i in range(len(u))])
        J+=price

        x = model.f(x, u, True, k)
        trajectory.append(x) # update the list of the trajectory
        actions.append(u) # update the list of the actions

    J += model.gN(x)
    return J, trajectory, actions, np.array(surpluses)

def tup_add(tup1,tup2):
    """
    Adds tuples together, elementwise
    """
    res = [int((tup1[i]+tup2[i])*10)/10 for i in range(len(tup1))] 
    return tuple(res)

def correct_traj_acts(x0,traj,acts):
    """
    Makes trajectory start with x0 and applies the actions to it
    
    Used in DP_P2P after running actions on ints to correct ints
    initial state to be the actual state
    """
    traj[0]=x0
    for i in range(len(acts)):
        traj[i+1]=tup_add(traj[i],acts[i])

    return traj

def _model_choice(model_name):
    """
    Function that returns the DPModel corresponding with the input
    
    Return type: class name, either DPModel, DPModel_c, or DPModel_both
    
    Usage: Used internally in _DP so as to use it for any model
    
    
    Input:
    
    model_name: str, should start with either p, c, or b
    
                If first letter is "p", returns price optimization DPModel
                If first letter is "c", returns emissions optimization DPModel_c
                If first letter is "b", returns the ratio weighted price and 
                emissions optimization DPModel_both
    
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    if model_name.lower()[0]=="p":
        return DPModel
    elif model_name.lower()[0]=="c":
        return DPModel_c
    elif model_name.lower()[0]=="b":
        return DPModel_both
    
    raise Exception("Input must be either 'price', 'carbon', or 'both'!!")
    
def _DP(model_name,Start,End,merged, battery,byday,ints,degrade,verbose,ratio):
    """
    Runs DP model that optimizes for either price, emissions, or both based
    on model_name
    
    Return type: Pandas dataframe
    
    Usage: Internal function that's used at both DP, DP_carb, and DP_both
    
    
    Input:
    
    model_name: str, should start with either p, c, or b
    
                If first letter is "p", runs price optimization
                If first letter is "c", runs emissions optimization
                If first letter is "b", runs the ratio weighted price 
                and emissions optimization
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
         
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
            
            Alternatively, this could be predictions, in which case it
            assumes that the predictions are the true values
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
    
    byday: bool, Flag that tells the function whether to run sequential DP models of size 24 hours
    
           This speeds the problem up by a lot, slightly less optimal 
    
    ints: bool, Flag that tells the DP model whether first look at a smaller problem
    
          It first runs a DP model with the ints flag, meaning it searches a smaller problem
          It then runs a second DP model, using the actions from the first to limit the 
          action space at each timestep
          
          This speeds the problem up by a lot, slightly less optimal 
    
    degrade: bool, Flag that tells function whether battery should degrade when charging battery
            
             Normally no degrading is done
             
    verbose: bool, Flag that tells function whether to print information about what it's doing
    
             Normally no printing is done
             
    ratio: float, if first letter of model_name is "b", only then is this used
    
           Sets the ratio used by the price and emissions optimization DP model
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    model = _model_choice(model_name)
    N=len(pd.date_range(start=Start,end=End,freq="h"))
    
    battery_ints = deepcopy(battery)
    
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
                DP_ints = model(Start_i, End_i, merged, deepcopy(battery_ints),degrade=degrade,ints=True)
            else:
                DP_ints = model(Start_i, End_i, merged, deepcopy(battery_ints),degrade=degrade,ints=True,ratio=ratio)
            _, pi_ints = DP_stochastic(DP_ints)
            _, _, actions_ints = policy_rollout(DP_ints,pi=lambda x, k: pi_ints[k][x],x0=int(battery_ints.get_current_capacity()))
            charge_i = list(actions_ints["charge"])
            
            if model_name.lower()[0]!="b":
                DP = model(Start_i, End_i, merged, deepcopy(battery), degrade=degrade, acts=charge_i, acts_range=1.5)
            else:
                DP = model(Start_i, End_i, merged, deepcopy(battery), degrade=degrade, acts=charge_i, acts_range=1.5, ratio=ratio)

        else:
            if model_name.lower()[0]!="b":
                DP = model(Start_i, End_i, merged,deepcopy(battery), degrade=degrade)
            else:
                DP = model(Start_i, End_i, merged,deepcopy(battery), degrade=degrade, ratio=ratio)
        
        _, pi = DP_stochastic(DP)
        _, _, actions = policy_rollout(DP,pi=lambda x, k: pi[k][x],x0=battery.get_current_capacity())
        series_battery_DP_i  = action_rollout(merged.loc[Start_i:End_i], battery, actions)

        series_battery_DP = pd.concat([series_battery_DP,series_battery_DP_i])
        
        Start_i= pd.date_range(start=End_i,periods=2,freq="h")[-1]

    series_battery_DP["cost_cummulative"] = series_battery_DP["cost"].cumsum(axis=0)
    series_battery_DP["emission_cummulative"] = series_battery_DP["emission"].cumsum(axis=0) 
    
    return series_battery_DP
    
def DP(Start,End,merged,battery,byday=True,ints=True,degrade=False,verbose=True):
    """
    Runs DP model that optimizes for price and rolls out its actions onto merged
    
    Return type: Pandas dataframe
    
    Usage: The "official" DP price optimization function
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
         
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
            
            Alternatively, this could be predictions, in which case it
            assumes that the predictions are the true values
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
    
    byday: bool, Flag that tells the function whether to run sequential DP models of size 24 hours
    
           By default this is True, meaning it runs sequential DP models each of size 24 hours
    
           This speeds the problem up by a lot, slightly less optimal 
    
    ints: bool, Flag that tells the DP model whether first look at a smaller problem
    
          It first runs a DP model with the ints flag, meaning it searches a smaller problem
          It then runs a second DP model, using the actions from the first to limit the 
          action space at each timestep
          
          By default, this is True, meaning it first looks at a smaller problem
          
          This speeds the problem up by a lot, slightly less optimal 
    
    degrade: bool, Flag that tells function whether battery should degrade when charging battery
            
             By default, this is False, meaning no degrading is done
             
    verbose: bool, Flag that tells function whether to print information about what it's doing
    
             By default, this is False, meaning no printing is done

    
    Example: merged = merge("h16")
            
             rf = RF(house)
            
             pred = rf.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")
            
             actions = DP("2022-06-19 00:00:00","2022-06-19 23:00:00",pred,Battery(max_capacity=13),
                          byday=True,ints=True,degrade=False,verbose=False)
    """
    return _DP("p",Start,End,merged,battery,byday,ints,degrade,verbose,None)

def DP_carb(Start,End,merged,battery,byday=True,ints=True,degrade=False,verbose=True):
    """
    Runs DP model that optimizes for emissions and rolls out its actions onto merged
    
    Return type: Pandas dataframe
    
    Usage: The "official" DP price optimization function
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
         
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
            
            Alternatively, this could be predictions, in which case it
            assumes that the predictions are the true values
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
    
    byday: bool, Flag that tells the function whether to run sequential DP models of size 24 hours
    
           By default this is True, meaning it runs sequential DP models each of size 24 hours
    
           This speeds the problem up by a lot, slightly less optimal 
    
    ints: bool, Flag that tells the DP model whether first look at a smaller problem
    
          It first runs a DP model with the ints flag, meaning it searches a smaller problem
          It then runs a second DP model, using the actions from the first to limit the 
          action space at each timestep
          
          By default, this is True, meaning it first looks at a smaller problem
          
          This speeds the problem up by a lot, slightly less optimal 
    
    degrade: bool, Flag that tells function whether battery should degrade when charging battery
            
             By default, this is False, meaning no degrading is done
             
    verbose: bool, Flag that tells function whether to print information about what it's doing
    
             By default, this is False, meaning no printing is done

    
    Example: merged = merge("h16")
            
             rf = RF(house)
            
             pred = rf.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")
            
             actions = DP_carb("2022-06-19 00:00:00","2022-06-19 23:00:00",pred,Battery(max_capacity=13),
                               byday=True,ints=True,degrade=False,verbose=False)
    """
    return _DP("c",Start,End,merged,battery,byday,ints,degrade,verbose,None)

def DP_both(Start,End,merged,battery,byday=True,ints=True,degrade=False,verbose=True,ratio=0.5):
    """
    Runs DP model that optimizes for price and rolls out its actions onto merged
    
    Return type: Pandas dataframe
    
    Usage: The "official" DP price optimization function
    
    
    Input:
    
    Start: str or Pandas Timestamp, Defines the starting date of the model
            
           Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
           If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
    End: str or Pandas Timestamp, Defines the end date of the model
            
         Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
         If timestamp: pd.Timestamp("2020-12-22 00:00:00")
         
    merged: Pandas dataframe, Raw data; output of "merged" function
            
            Needed for yield values, indices, and spotprices
            
            Alternatively, this could be predictions, in which case it
            assumes that the predictions are the true values
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
    
    byday: bool, Flag that tells the function whether to run sequential DP models of size 24 hours
    
           By default this is True, meaning it runs sequential DP models each of size 24 hours
    
           This speeds the problem up by a lot, slightly less optimal 
    
    ints: bool, Flag that tells the DP model whether first look at a smaller problem
    
          It first runs a DP model with the ints flag, meaning it searches a smaller problem
          It then runs a second DP model, using the actions from the first to limit the 
          action space at each timestep
          
          By default, this is True, meaning it first looks at a smaller problem
          
          This speeds the problem up by a lot, slightly less optimal 
    
    degrade: bool, Flag that tells function whether battery should degrade when charging battery
            
             By default, this is False, meaning no degrading is done
             
    verbose: bool, Flag that tells function whether to print information about what it's doing
    
             By default, this is False, meaning no printing is done

    ratio: float, How much to weight emissions compared to spotprices
    
           Should be between 0 and 1
           
           Used the following way:
           (1-ratio)*spotprices + ratio*emissions
    
    
    Example: merged = merge("h16")
            
             rf = RF(house)
            
             pred = rf.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")
            
             actions = DP_both("2022-06-19 00:00:00","2022-06-19 23:00:00",pred,Battery(max_capacity=13),
                               byday=True,ints=True,degrade=False,verbose=False)
    """
    return _DP("b",Start,End,merged,battery,byday,ints,degrade,verbose,ratio)
   
class DP_P2P:
    def __init__(self, start_time, end_time, merges, houses, battery):
        if len(houses)!= len(set(houses)):
            raise Exception("Duplicate houses are not allowed! Only one type of each house!")
        if len(houses)<=1:
            raise Exception("P2P requires more than one house!")
        if False in [house in ["k28", "h16", "h22", "h28", "h32"] for house in houses]:
            raise Exception('All houses should be either "k28", "h16", "h22", "h28", or "h32"')
        if type(battery) is not Battery:
            raise Exception("battery must be a Battery class instance!")
            
        #Attributes, needed for functions
        self.start_time = start_time
        self.end_time = end_time
        self.merges = merges
        self.houses = houses
        self.battery = battery
        self.N = len(pd.date_range(start=start_time,end=end_time,freq="h"))
        
        #For P2P_sol function
        self.results = None
        self.results_ord = None
        
        #For cost_matrix function
        self.sp = merges[0].loc[start_time:end_time]["SpotPriceDKK"]/1000
        self.ep = merges[0].loc[start_time:end_time]["CO2Emission"]/1000 #Only the first element in merges needs carbon predictions
        self.nf = None
        self.split_costs = None
        self.split_emis  = None
        
        #For all_sol function
        self.all_results = None
        self.all_results_ord = None
        self.all_nf = None
        self.all_costs = None
        self.best_idx = None
        
        
        #Ordered houses and merges
        all_houses = ["h16", "h22", "h28", "h32", "k28"] 
        self.all_houses = all_houses
        
        houses_ord = []
        merges_ord = []
        i=0
        for house in self.all_houses:
            if house in self.houses:
                houses_ord.append(house)
                merges_ord.append(merges[i])
                i+=1
                
        self.houses_ord = houses_ord
        self.merges_ord = merges_ord

        #Permutation attributes
        perms = list(permutations([i+1 for i in range(len(self.houses))]))
        
        houses_rewrites = [[self.houses_ord[i-1] for i in perms[j]] for j in range(len(perms))]
        merges_rewrites = [[self.merges_ord[i-1] for i in perms[j]] for j in range(len(perms))]
        
        
        perms_names = [''.join(str(i) for i in perms[j]) for j in range(len(perms))]
        
        self.perms = perms
        self.houses_rewrites = houses_rewrites
        self.merges_rewrites = merges_rewrites
        self.perms_names = perms_names
        
        self.this_idx, self.this_perm, self.this_perm_name = self.find_this_perm()
        
        #For find_constraints function
        self.constraints = None
        self.constraints_ord = None
        self.all_series = None
        self.all_series_ord = None

    def find_constraints(self, x0, verbose=False):
        """
        Returns the constraints in the order of the current permutation (this_perm)
        """
        #If constraints have NOT previously been calculated
        if self.constraints_ord is None:
            constraints = []
            all_series = []
            
            #Runs single house opt DP on each of the houses to get constraints
            for i in range(len(self.houses)):
                bat_copy = deepcopy(self.battery)
                bat_copy.current_capacity = x0[i]
                series = DP(self.start_time, self.end_time, self.merges[i], bat_copy, 
                            byday=True, ints=True, degrade=False, verbose=False)
                constraints.append(series["cost"].sum())
                all_series.append(series)

            self.constraints = constraints
            self.all_series  = all_series

            #Ordered constraints
            constraints_ord = list(self.this_perm)
            all_series_ord = deepcopy(all_series)
            for j, i in enumerate(self.this_perm):
                constraints_ord[i-1] = constraints[j]
                all_series_ord[i-1] = all_series[j]

            self.constraints_ord = constraints_ord
            self.all_series_ord  = all_series_ord
            
        #If constraints HAVE previously been calculated
        else:
            constraints = [self.constraints_ord[i-1] for i in self.this_perm]
            self.constraints = constraints
        
        if verbose:
            print(f"The optimality constraints are as follows:")
            toprint = [house + ": " + str(constraints[i]) for i, house in enumerate(self.houses)]
            print(*toprint, sep=", ")
        
    def find_this_perm(self):
        """
        Calculates the current permutation
        """
        #Caclulates the current permutation
        for i, houses in enumerate(self.houses_rewrites):
            if tuple(houses)==tuple(self.houses):
                this_idx = i
                this_perm = self.perms[this_idx]
                this_perm_name = self.perms_names[this_idx]   
                
                return this_idx, this_perm, this_perm_name
    
    def DP(self, j, Start, End, x0, max_number_states, trajectory=None, actions=None, ints=False):
        """
        Runs the DP model for P2P. This is made specifically for use in P2P_sol function, so don't use this
        directly
        """
        #This is usually the very first run and is, therefore, only really used once per P2P_sol
        if ints:   
            #Makes an integer initial state, so that x0_int is in the state space
            x0_int = tuple([int(x0[i]) for i in range(len(x0))])
            
            #This makes sure we only search for actions in the first house
            furthers=[[1,0]]
            for t in range(len(self.houses)-1):
                furthers.append([self.battery.max_capacity+1,x0_int[t+1]])

            #DP and rollout
            DPP2P= DP_central(Start, End, self.merges, self.houses, deepcopy(self.battery),
                              degrade=False,ints=True,acts=None,acts_range=None,
                              furthers=furthers,traj=None,traj_range=None, max_number_states=max_number_states)
            _, pi = DP_stochastic(DPP2P)
            
            J, trajectory, actions, surpluses = p2p_rollout(model=DPP2P, pi=lambda x, k: pi[k][x], x0=x0_int)

            #Corrects the integer 
            if x0 != x0_int:
                trajectory = correct_traj_acts(x0,trajectory,actions)

                
        #After having some initial actions from ints run, this gets run
        elif (trajectory is not None) and (actions is not None):
            
            #This makes sure we only search for actions for one house at a time, that being house number j
            traj_range = [0 if t!=j+1 else -1 for t in range(len(self.houses))]  
            
            #This make sure we search close to previously found actions
            acts_range = [0 if t!=j+1 else self.battery.max_charge for t in range(len(self.houses))]
            
            #DP and rollout
            DPP2P= DP_central(Start, End, self.merges, self.houses, deepcopy(self.battery),
                              degrade=False,ints=False,acts=actions,acts_range=acts_range,
                              furthers=None,traj=trajectory,traj_range=traj_range, max_number_states=max_number_states)
            _, pi = DP_stochastic(DPP2P)

            J, trajectory, actions, surpluses = p2p_rollout(model=DPP2P, pi=lambda x, k: pi[k][x], x0=x0)

        else:
            raise Exception("Must have either furthers or trajectory and actions as input!")

        return J, trajectory, actions, surpluses
        
    def P2P_sol(self, x0, max_number_states=20, byday=True, verbose=True):
        """
        Finds the P2P solution for this permutation only. Perhaps use all_sol function?
        """
        #Pre-loop settings
        N=self.N
        all_actions = []
        all_surpluses = np.ones((0,len(self.houses)))
        Start_i = self.start_time
        x0_i = x0
        num_loops = int(np.ceil(N/24)) if byday else 1
        remainder = N%24
        length = 24 if byday else N
        
        for i in range(num_loops):
            if byday and i == num_loops-1:
                length = length if remainder == 0 else remainder

            End_i = pd.date_range(start=Start_i,periods=length,freq="h")[-1]

            if verbose:
                print(f"Period from {Start_i} to {End_i}")
            
            for j in range(len(self.houses)):
                #Initial ints run
                if j==0:
                    J, trajectory, actions, surpluses = self.DP(j, Start_i, End_i, x0_i, max_number_states, 
                                                                 trajectory=None, actions=None, ints=True)
                #Runs for one house at a time        
                J, trajectory, actions, surpluses = self.DP(j, Start_i, End_i, x0_i, max_number_states, 
                                                             trajectory=trajectory, actions=actions, ints=False)
            #Update results
            all_actions = all_actions + actions
            all_surpluses = np.append(all_surpluses,surpluses,axis=0)

            #End-loop settings
            Start_i= pd.date_range(start=End_i,periods=2,freq="h")[-1]
            x0_i = trajectory[-1]

        #Orders results
        all_surpluses_ord = np.copy(all_surpluses)
        all_actions_ord = np.array(all_actions)
        
        _,this_perm,_ = self.find_this_perm()
        
        for j, i in enumerate(this_perm):
            all_surpluses_ord[:,i-1] = all_surpluses[:,j]
            all_actions_ord[:,i-1] = np.array(all_actions)[:,j]
        

        all_actions_ord = list(map(tuple,all_actions_ord))
        
        #Save results
        self.results = (all_actions,all_surpluses)
        self.results_ord =(all_actions_ord,all_surpluses_ord)
        
        self.cost_matrix()
        
        return self.results_ord
    
    def cost_matrix(self):
        """
        Calculates the ordered P2P cost matrix for the saved solution
        If no solution is saved, returns None
        """
        
        if self.results_ord is None:
            print("results_ord not found. Try running P2P_sol or all_sol")
            return None
        
        #Cost matrix will always be in sorted order
        _,surpluses_ord = self.results_ord
        
        #Surpluses needed to calculate costs using EnergyMarket class
        nf = pd.DataFrame()
        for i,house in enumerate(self.houses_ord):
            nf[house] = surpluses_ord[:,i]
            
        #Calculate costs and emissions
        costs = pd.DataFrame()
        emissions = pd.DataFrame()
        split_costs = []
        split_emis  = []
        for i in range(len(nf)):
            em = EnergyMarket(nf.iloc[i].to_dict(),self.sp[i],self.sp[i]+1)
            temp = pd.DataFrame(em.get_total_costs(), index=[0])
            costs = pd.concat([costs,temp],ignore_index=True)
            
            split_costs.append(em.split_costs)
            
            em = EnergyMarket(nf.iloc[i].to_dict(),-1,self.ep[i],True)
            temp = pd.DataFrame(em.get_total_costs(), index=[0])
            emissions = pd.concat([emissions,temp],ignore_index=True)
            
            split_emis.append(em.split_costs)
        
        self.split_costs = split_costs
        self.split_emis  = split_emis
            
        #Several loops so columns appear in easier to read order
        for house in self.houses_ord:
            nf['cost_'+house] = costs[house].to_list()
        for house in self.houses_ord:
            nf['cumm_cost_'+house] = nf['cost_'+house].cumsum()  
        for house in self.houses_ord:
            nf['emis_'+house] = emissions[house].to_list()
        for house in self.houses_ord:
            nf['cumm_emis_'+house] = nf['emis_'+house].cumsum()
         
        self.nf = nf
        return nf
    
    def total_cost(self):
        """
        Total cost of the current solution if any, otherwise None
        """
        if self.nf is None:
            print("nf not found. Try running P2P_sol or all_sol")
            return 
        
        return sum([self.nf['cumm_cost_'+house][len(self.nf)-1] for house in self.houses])
    
    def all_sol(self, x0, max_number_states=20, stop_early=True, byday=True, verbose=True):
        """
        The definitive run to find a solution that fulfills the optimality constraints
        
        If it doesn't find one, all solutions it found are saved, otherwise only the
        first solution that fulfills constraints is saved
        """
        #Constructs the ordered initial state
        x0_ord = list(self.this_perm)
        for j, i in enumerate(self.this_perm):
            x0_ord[i-1] = x0[j]
            
        x0_ord = tuple(x0_ord)
        
        
        #Before loop
        all_results = []
        all_results_ord = []
        all_nf = []
        all_costs = []
            
        if verbose:
            print(f"This permutation is called '{self.this_perm_name}'. Starting with permutation '{self.perms_names[0]}'") 
            
        for i in range(len(self.houses_rewrites)):
            fulfills = False
            if verbose:
                print(f"Permutation {self.perms_names[i]}, ({i+1}/{len(self.perms_names)})")
            
            #Updates parameters for use in class functions calls
            x0 = tuple([x0_ord[j-1] for j in self.perms[i]])
            self.houses = self.houses_rewrites[i]
            self.merges = self.merges_rewrites[i]
            self.this_perm = self.perms[i]
            
            if verbose:
                print()
            self.find_constraints(x0, verbose = verbose)
            
            if verbose:
                print()
            
            #Finds current permutation's solution and cost matrix
            self.P2P_sol(x0, max_number_states=max_number_states, byday=byday, verbose=verbose)
            
            #Updates
            all_results.append(self.results)
            all_results_ord.append(self.results_ord)
            all_nf.append(self.nf)
            all_costs.append(self.total_cost())
                  
            #Checks if constraints are fulfilled
            checks = [self.nf['cumm_cost_'+house][len(self.nf)-1]<=self.constraints[i] for i,house in enumerate(self.houses)]
            if not (False in checks):
                fulfills = True
                if verbose:
                    print(f"Permutation {self.perms_names[i]} fulfills constraints!")
            
            #Stopping early if constraints are fulfilled
            if fulfills and stop_early:  
                if verbose:
                    print("Stopping early...")
                    
                #Resets parameters to what they were before the loop
                self.houses = self.houses_rewrites[self.this_idx]
                self.merges = self.merges_rewrites[self.this_idx]
                self.this_perm = self.perms[self.this_idx]
                x0 = tuple([x0_ord[j-1] for j in self.perms[self.this_idx]])
                self.find_constraints(x0)
                
                self.update_series(x0_ord)
                return
                
            if verbose:
                print()
                print()
        
        #Resets parameters to what they were before the loop
        self.houses = self.houses_rewrites[self.this_idx]
        self.merges = self.merges_rewrites[self.this_idx]
        self.this_perm = self.perms[self.this_idx]
        x0 = tuple([x0_ord[j-1] for j in self.this_perm])
        self.find_constraints(x0)      
        
        self.update_series(x0_ord)
        
        #Saves results
        self.all_results = all_results
        self.all_results_ord = all_results_ord
        self.all_nf = all_nf
        self.all_costs = all_costs
        self.best_idx = np.argmin(all_costs)
        
        if verbose:
            print("Done!")
            print()
            print(f"Best found permutation was {self.perms_names[self.best_idx]} with cost {self.all_costs[self.best_idx]}")
           
    def update_series(self, x0_ord):
        if (self.nf is None) or (self.all_series_ord is None) or (self.results_ord is None):
            print("nf, all_series_ord, or results_ord not found. Try running all_sol")
            return
        
        for i,house in enumerate(self.houses_ord):
            series = self.all_series_ord[i]
            acts,_ = self.results_ord
            
            actions = pd.DataFrame(columns=["charge"])
            actions["charge"]=np.array(acts)[:,i]
            actions.index = series.index
            
            bat_copy = deepcopy(self.battery)
            bat_copy.current_capacity = x0_ord[i]
            
            series = action_rollout(series,bat_copy,actions)
            
            all_grid = []
            all_peer = []
            for j in range(len(self.split_costs)):
                grid,peer = self.split_costs[j][house]
                
                all_grid.append(grid)
                all_peer.append(peer)
                
            series["grid"] = all_grid
            series["peer"] = all_peer
            
            series["cost"]                 = list(self.nf['cost_'+house]) 
            series["cost_cummulative"]     = list(self.nf['cumm_cost_'+house])
            series["emission"]             = list(self.nf['emis_'+house]) 
            series["emission_cummulative"] = list(self.nf['cumm_emis_'+house])
            
            self.all_series_ord[i] = series
            
        self.all_series = [self.all_series_ord[j-1] for j in self.this_perm]
        
    def sol_print(self):
        """
        Prints the solution obtained from P2P_sol or all_sol
        """
        if self.nf is None:
            print("nf not found. Try running P2P_sol or all_sol")
            return
        
        display(HTML(self.nf._repr_html_()))
        
    def all_sol_print(self):
        """
        Mostly used to see what solutions the all_sol spits out when it doesN'T stop early
        """
        if self.all_nf is None:
            print("all_nf not found. Try running all_sol with stop_early=False")
            return
        
        ognf = self.nf
        for i in range(len(stuff.all_nf)):
            self.nf = self.all_nf[i]
            print(f"This is the cost matrix for sol with permutation {self.perms_names[i]}")
            self.sol_print()
            print(f"Total cost this permutation = {self.all_costs[i]}")
            print()
            print()
            print()
            
        #Resets parameter to what it was before the loop
        self.nf = ognf
            
    def all_sol_actions(self):
        """
        Mostly used to see what actions the all_sol spits out when it doesN'T stop early
        """
        if self.all_results_ord is None:
            print("all_results_ord not found. Try running all_sol with stop_early=False")
            return
        df = pd.DataFrame()
        for i in range(len(self.all_results_ord)):
            df["actions"+self.perms_names[i]] = self.all_results_ord[i][0]
        return df
    
    def series_print(self, house):
        if house not in self.houses:
            raise Exception(f"Input house must be contained in this instance's houses: {self.houses}")
        
        if (self.all_series is None):
            print("all_series not found. Try running all_sol")
            return
            
        i = self.houses.index(house)
            
        logic_series_print(self.all_series[i], p2p=True)
        
    def all_series_print(self):
        for house in self.houses_ord:
            print(f"Series for house {house}:")
            self.series_print(house)
            print()
            print()
    
if __name__ == "__main__":
    print("This file is meant to be imported")
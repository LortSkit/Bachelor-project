import numpy as np
import pandas as pd
from Logic import get_price, get_emissions, action_rollout
from copy import deepcopy

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
        
        return get_price(yieldd-charge,self.sp[k],0.1)
    
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
    
if __name__ == "__main__":
    print("This file is meant to be imported")
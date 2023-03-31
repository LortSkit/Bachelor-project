import pandas as pd
from dateutil.relativedelta import relativedelta
from copy import deepcopy
from Logic import logic_actions, logic_bat

def get_price(surplus, spot_price, percentage_cut):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    surplus: (type), (Explanation)
             
             (Explanation)
    
    spot_price: (type), (Explanation)
             
                (Explanation)
                
    percentage_cut: (type), (Explanation)
             
                    (Explanation)
                    
    
    Example: get_price(-5.5, 0.154039, 0.1) = 0.8472145
    '''
    
    #Sell
    if surplus > 0:
        return -surplus * spot_price*percentage_cut
    #Buy
    else:
        return -surplus * spot_price

    
def get_emissions(surplus, emission):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    surplus: (type), (Explanation)
             
             (Explanation)
    
    emission: (type), (Explanation)
             
              (Explanation)
                    
    
    Example: get_emissions(-5.5, 0.1370) = 0.7535
    '''
    
    return -surplus*emission


def logic_rollout(series_battery, battery, logic, actions=None):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    series_battery: (type), (Explanation)
             
                    (Explanation)
    
    battery: (type), (Explanation)
             
             (Explanation)
             
    logic: (type), (Explanation)
             
           (Explanation)
             
    actions: (type), (Explanation)
             
             (Explanation)
             
    
    Example: logic_rollout(merged.loc[Start:End], Battery(max_capacity=13), logic_bat)
    '''
    
    series_battery = series_battery.apply(lambda row: logic(row, battery, actions), axis=1)
    series_battery["cost"] = series_battery.apply(lambda row: get_price(row["surplus"], row["SpotPriceDKK"]/1000,0.1), axis=1)
    series_battery["emission"] = series_battery.apply(lambda row: get_emissions(row["surplus"],row["CO2Emission"]/1000), axis=1)
    series_battery["cost_cummulative"] = series_battery["cost"].cumsum(axis=0)
    series_battery["emission_cummulative"] = series_battery["emission"].cumsum(axis=0)
    return series_battery

def action_rollout(series_battery, battery, actions):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    series_battery: (type), (Explanation)
             
                    (Explanation)
    
    battery: (type), (Explanation)
             
             (Explanation)
             
    actions: (type), (Explanation)
             
             (Explanation)
             
    
    Example: action_rollout(merged_i.iloc[:length], Battery(max_capacity=13), actions[:length])
    '''
    return logic_rollout(series_battery, battery, logic_actions, actions)
    

def pred_logic_rollout(series_battery_true,series_battery_pred, battery, logic):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    series_battery_true: (type), (Explanation)
             
                         (Explanation)
    
    series_battery_pred: (type), (Explanation)
             
                         (Explanation)
    
    battery: (type), (Explanation)
             
             (Explanation)
             
    logic: (type), (Explanation) REDACTED, NOT IN USE
             
           (Explanation)
             
    
    Example: pred_logic_rollout(merged_i, pred_i, Battery(max_capacity=13), logic_bat)
    '''
    
    series_battery_pred = logic_rollout(series_battery_pred, deepcopy(battery), logic_bat, None)
    
    series_battery_true = action_rollout(series_battery_true, battery, series_battery_pred)
    
    return series_battery_true
    
    
def print_price_summary(series_battery,yearprint=True):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    series_battery: (type), (Explanation)
             
                    (Explanation)
             
    
    Example: print_price_summary(logic_rollout(merged.loc[Start:End], Battery(max_capacity=13), logic_bat))
    '''
    
    start, end = series_battery.index[0], series_battery.index[-1]
    difference_in_years = relativedelta(end, start).years
    print(f"The period is from {start} to {end}")
    print(f"Cost for period: ", round(series_battery["cost"].sum(), 0), " DKK")
    print(f"Total emissions for period: ", round(series_battery["emission"].sum(),0), " kg")
    num_wh_total = series_battery[series_battery["surplus"] < 0]["surplus"].sum()  
    num_wh_total_sold = series_battery[series_battery["surplus"] > 0]["surplus"].sum() 

    time_delta_seconds =  (end-start).total_seconds()
    years_timedelta = time_delta_seconds/(365.25*24*60*60)
    
    if yearprint:
        print(f"Average cost per year is: {round(series_battery['cost'].sum()/years_timedelta,0)} DKK")
        print(f"Average emissions per year is: {round(series_battery['emission'].sum()/years_timedelta,0)} kg")

    print(f"Number of kwh purchased in the period: {-num_wh_total}")

    if yearprint:
        print(f"Average number of kwh purchased per year: {-num_wh_total/years_timedelta}")
       
        print(f"Average number of kwh sold per year: {num_wh_total_sold/years_timedelta}")
        


    return round(series_battery['cost'].sum()/years_timedelta,0),-num_wh_total/years_timedelta, num_wh_total_sold/years_timedelta


def logic_series_print(series_battery):
    '''
    (Description)
    
    Usage: (Explanation)
    
    
    Input:
    
    series_battery: (type), (Explanation)
             
                    (Explanation)
             
    
    Example: logic_series_print(logic_rollout(merged.loc[Start:End], Battery(max_capacity=13), logic_bat))
    '''
    
    print(f"{'hour':8s} {'price':8s} {'eprice':8s} {'yield':8s} {'surplus':8s} {'buy':8s} {'charge':8s} {'before':8s} {'degrade':8s} {'after':8s} {'cost':8s} {'pcumsum':8s} {'emis':8s} {'ecumsum':8s}")

    for i in range(len(series_battery)):
        spot    = series_battery.iloc[i]['SpotPriceDKK']/1000
        eprice  = series_battery.iloc[i]['CO2Emission']/1000
        yieldd  = series_battery.iloc[i]['yield']
        surplus = series_battery.iloc[i]['surplus']
        buy     = series_battery.iloc[i]['buy']
        charge  = series_battery.iloc[i]['charge']
        before  = series_battery.iloc[i]['capacity_before']
        degrade = series_battery.iloc[i]['capacity_degraded']
        after   = series_battery.iloc[i]['capacity_after']
        cost    = series_battery.iloc[i]['cost']
        emis    = series_battery.iloc[i]['emission']
        cost_c  = series_battery.iloc[i]['cost_cummulative']
        emis_c  = series_battery.iloc[i]['emission_cummulative']
        print(f"{i:5d}: {spot:8.4f},{eprice:8.4f},{yieldd:8.4f},{surplus:8.4f},{buy:8.4f},{charge:8.4f},{before:8.4f},{degrade:8.4f},{after:8.4f},{cost:8.4f},{cost_c:8.4f},{emis:8.4f},{emis_c:8.4f}")
        
        
def policy_rollout(model, pi, x0):
    """
    Given an environment and policy, should compute one rollout of the policy and compute
    cost of the obtained states and actions. In the deterministic case this corresponds to

    J_pi(x_0)

    in the stochastic case this would be an estimate of the above quantity.

    Note I am passing a policy 'pi' to this function. The policy is in this case itself a function, that is,
    you can write code such as

    > u = pi(x,k)

    in the body below.
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


def DP_stochastic(model):
    """
    Implement the stochastic DP algorithm. The implementation follows (Her21, Algorithm 1).
    In case you run into problems, I recommend following the hints in (Her21, Subsection 6.2.1) and focus on the
    case without a noise term; once it works, you can add the w-terms. When you don't loop over noise terms, just specify
    them as w = None in env.f and env.g.
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        #print(f"\r{k} ",end="")
        for x in model.S(k): 
            """
            Update pi[k][x] and Jstar[k][x] using the general DP algorithm given in (Her21, Algorithm 1).
            If you implement it using the pseudo-code, I recommend you define Q as a dictionary like the J-function such that
                        
            > Q[u] = Q_u (for all u in model.A(x,k))
            Then you find the u where Q_u is lowest, i.e. 
            > umin = arg_min_u Q[u]
            Then you can use this to update J[k][x] = Q_umin and pi[k][x] = umin.
            """
            w = False

            Qu = {tuple(u): (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for u in model.A(x, k)} 
            umin = min(Qu, key=Qu.get)
            J[k][x] = Qu[umin]
            pi[k][x] = umin


            """
            After the above update it should be the case that:

            J[k][x] = J_k(x)
            pi[k][x] = pi_k(x)
            """
    return J, pi

if __name__ == "__main__":
    print("This file is meant to be imported")
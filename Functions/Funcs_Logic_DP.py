import pandas as pd
from dateutil.relativedelta import relativedelta
from copy import deepcopy
from Logic import logic_actions

# Definge get price function   
def get_price(demand, spot_price, percentage_cut):

    #Sell
    if demand > 0:
        return -demand * spot_price*percentage_cut #Price zone DK1
    #Buy
    else:
        return -demand * spot_price #Price zone DK1

#Logic functions
def logic_rollout(series_battery, battery, logic, actions=None):
    
    series_battery = series_battery.apply(lambda row: logic(row, battery, actions), axis=1)
    series_battery["price"] = series_battery.apply(lambda row: get_price(row["surplus"], row["SpotPriceDKK"]/1000,0.1), axis=1)
    series_battery["price_cummulative"] = series_battery["price"].cumsum(axis=0)
    return series_battery

def action_rollout(series_battery, battery, actions):
    
    series_battery = series_battery.apply(lambda row: logic_actions(row,battery,actions), axis=1)
    series_battery["price"] = series_battery.apply(lambda row: get_price(row["surplus"], row["SpotPriceDKK"]/1000,0.1), axis=1)
    series_battery["price_cummulative"] = series_battery["price"].cumsum(axis=0)
    return series_battery
    

def pred_logic_rollout(series_battery_true,series_battery_pred, battery, logic, actions=None):
    
    series_battery_pred = logic_rollout(series_battery_pred, deepcopy(battery), logic, actions)
    
    series_battery_true = action_rollout(series_battery_true, battery, series_battery_pred)
    return series_battery_true
    
def print_price_summary(series_battery):
    start, end = series_battery.index[0], series_battery.index[-1]
    difference_in_years = relativedelta(end, start).years
    print(f"Cost for period: {start} to {end} is: ", round(series_battery["price"].sum(), 0), " DKK")
    num_wh_total = series_battery[series_battery["surplus"] < 0]["surplus"].sum()  
    num_wh_total_sold = series_battery[series_battery["surplus"] > 0]["surplus"].sum() 

    time_delta_seconds =  (end-start).total_seconds()
    years_timedelta = time_delta_seconds/(365.25*24*60*60)
    print(f"Average cost per year is: {round(series_battery['price'].sum()/years_timedelta,0)} DKK")

    print(f"Number of kwh purchased in the period: {-num_wh_total}")

    print(f"Average number of kwh purchased per year: {-num_wh_total/years_timedelta}")
       
    print(f"Average number of kwh sold per year: {num_wh_total_sold/years_timedelta}")


    return round(series_battery['price'].sum()/years_timedelta,0),-num_wh_total/years_timedelta, num_wh_total_sold/years_timedelta

#Prints the all action sequences (along with other goodies) using the series_battery attained from logic_rollout
def logic_series_print(series_battery):
    print(f"{'hour':8s} {'price':8s} {'yield':8s} {'surplus':8s} {'buy':8s} {'charge':8s} {'before':8s} {'after':8s} {'cost':8s} {'cumsum':8s}")

    for i in range(len(series_battery)):
        spot    = series_battery.iloc[i]['SpotPriceDKK']/1000
        yieldd  = series_battery.iloc[i]['power_yield']
        surplus = series_battery.iloc[i]['surplus']
        buy     = series_battery.iloc[i]['buy']
        charge  = series_battery.iloc[i]['charge']
        before  = series_battery.iloc[i]['capacity_before']
        after   = series_battery.iloc[i]['capacity_after']
        cost    = series_battery.iloc[i]['price']
        cost_c  = series_battery.iloc[i]['price_cummulative']
        print(f"{i:5d}: {spot:8.4f},{yieldd:8.4f},{surplus:8.4f},{buy:8.4f},{charge:8.4f},{before:8.4f},{after:8.4f},{cost:8.4f},{cost_c:8.4f}")
        
        
#DP functions
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
    actions = pd.DataFrame(actions,columns=['charge','buy'])
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
        print(f"\r{k} ",end="")
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
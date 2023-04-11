import pandas as pd
from dateutil.relativedelta import relativedelta
from copy import deepcopy

def get_price(surplus, spot_price, percentage_cut):
    '''
    Returns cost of buying surplus (kWh) amount at price of spot_price (DKK/kWh)
    when surplus is negative, when positive returns the negative cost of selling 
    a surplus amount at percentage_cut (%) times the spot_price
    
    Return type: float
    
    Usage: Used in MPC and DP cost functions, used when rolling out logic or actions
    
    
    Input:
    
    surplus: float, the amount buying (negative) or the amount selling (positive)
             
             surplus = yield-charge, where yield = production - consumption and
             charge is the amount chosen to charge, both for a given timestep
             
             Both production, consumption and yield are "officially" obtained
             from the "merge" function from file "Merge.py"
    
    spot_price: float, the price of 1 kWh in DKK
             
                The spot prices are "officially" obtained from the "merge"
                function from file "Merge.py"
                
    percentage_cut: float, between 0 and 1, determines sell value buy percentage_cut*spot_price 
             
                    This is to emulate the fact that when selling to the grid, taxes
                    have to be paid, which means when percentage_cut=0.1 (report standard) then
                    there's 90% taxes
                    
    
    Example: get_price(-5.5, 0.154039, 0.1) #= 0.8472145
    '''
    
    #Sell
    if surplus > 0:
        return -surplus * spot_price*percentage_cut
    #Buy
    else:
        return -surplus * spot_price

    
def get_emissions(surplus, emission):
    '''
    Returns emission cost of buying surplus (kWh) amount at price of 
    emission (kg/kWh) when surplus is negative, when positive returns the 
    negative cost of selling a surplus amount at a gain of emission
    
    Return type: float
    
    Usage: Used in MPC and DP cost functions, used when rolling out logic or actions
    
    
    Input:
    
    surplus: float, the amount buying (negative) or the amount selling (positive)
             
             surplus = yield-charge, where yield = production - consumption and
             charge is the amount chosen to charge, both for a given timestep
             
             Both production, consumption and yield are "officially" obtained
             from the "merge" function from file "Merge.py"
    
    emission: float, the carbon emission price of 1 kWh in kg (of carbon)
             
                The emissions are "officially" obtained from the "merge"
                function from file "Merge.py"
                    
    
    Example: get_emissions(-5.5, 0.1370) #= 0.7535
    '''
    
    return -surplus*emission


def logic_bat(row, battery):
    """
    Simple logic used for rollout, used alongside Pandas DataFrame "apply" function
    
    Return type: Pandas series
    
    Usage: Used in internal function "_logic_rollout" for simple logic rollout
    
    
    Input:
    
    row: Pandas series, Assumed to be a row of a dataframe. Should have a "yield" column
            
         Should be used like series_battery.apply(lambda row: logic_bat(row, battery), axis=1)
         
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    yieldd = row["yield"]
    battery.charge(yieldd)
    
    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_degraded"] = battery.get_previous_degraded_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = battery.get_surplus()
    row["charge"] = battery.charge_list[-1]
    row["buy"] = 0.0
    return row


def logic_actions(row, battery, actions):
    """
    Action logic used for rollout, used alongside Pandas DataFrame "apply" function
    
    Return type: Pandas series
    
    Usage: Used in internal function "_logic_rollout" for action rollout
    
    
    Input:
    
    row: Pandas series, Assumed to be a row of a dataframe. Should have a "yield" column
            
         Should be used like series_battery.apply(lambda row: logic_actions(row, battery, actions), axis=1)
         
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    actions: Pandas dataframe, Should contain a "charge" column
             
             Actions are usually obtained from another rollout or optimization model. 
             Rollouts include: logic_rollout, actions_rollout, or pred_logic_rollout
             Optimization models include: DP and MPC (price, carb, both)
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    """
    yieldd = row["yield"]
    charge = actions.loc[row.name]["charge"]

    #Charging more than necessary
    if yieldd<=0:
        buy=charge if charge>0 else 0.0
    else:
        if yieldd<charge:
            buy=charge-yieldd
        else:
            buy=0.0

    #Discharging more than necessary
    if yieldd>0:
        sell=charge if charge<0 else 0.0
    else:
        if yieldd>=charge:
            sell=charge-yieldd
        else:
            sell=0.0

    battery.charge(charge)

    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_degraded"] = battery.get_previous_degraded_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = yieldd-charge
    row["charge"] = charge
    row["buy"] = buy+sell
    return row


def _logic_rollout(series_battery, battery, actions):
    '''
    Function that applies either logic or actions to series_battery using
    the battery
    
    Return type: Pandas dataframe
    
    Usage: Internal function that's used as both logic_rollout and action_rollout
    
    
    Input:
    
    series_battery: Pandas dataframe, Should contain a "yield" column
             
                    Usually obtained either from the raw data (perfect predictions)
                    or from a predictions dataframe
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    actions: Pandas dataframe, Should contain a "charge" column, or be None
             
             Actions are usually obtained from another rollout or optimization model. 
             Rollouts include: logic_rollout, actions_rollout, or pred_logic_rollout
             Optimization models include: DP and MPC (price, carb, both)
             
             If None, will run logic_rollout, otherwise runs action_rollout
             
    
    Example: THIS IS AN INTERNAL FUNCTION, SHOULD NOT BE IMPORTED
    '''
    
    if actions is not None:
        series_battery = series_battery.apply(lambda row: logic_actions(row, battery, actions), axis=1)
    else:
        series_battery = series_battery.apply(lambda row: logic_bat(row, battery), axis=1)
    
    fee = 1 #transmission fee
    series_battery["cost"] = series_battery.apply(lambda row: get_price(row["surplus"], fee+row["SpotPriceDKK"]/1000,0.1), axis=1)
    series_battery["emission"] = series_battery.apply(lambda row: get_emissions(row["surplus"],row["CO2Emission"]/1000), axis=1)
    series_battery["cost_cummulative"] = series_battery["cost"].cumsum(axis=0)
    series_battery["emission_cummulative"] = series_battery["emission"].cumsum(axis=0)
    return series_battery


def logic_rollout(series_battery, battery):
    '''
    Function that applies the simple logic used in the report. If
    the battery has max_capacity = 0, then this is also the "no
    battery" model.
    
    Return type: Pandas dataframe
    
    Usage: Used when wanting to simulate perfect predictions
    
    
    Input:
    
    series_battery: Pandas dataframe, Should contain a "yield" column
             
                    Usually obtained either from the raw data (perfect predictions)
                    or from a predictions dataframe
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    
    Example: merged = merge("h16")
    
             series = logic_rollout(merged.loc[Start:End], Battery(max_capacity=13))
    '''
    return _logic_rollout(series_battery, battery, None)


def action_rollout(series_battery, battery, actions):
    '''
    Function that applies the input actions.
    
    Return type: Pandas dataframe
    
    Usage: Used when getting actions from other series or any non-logic model
    
    
    Input:
    
    series_battery: Pandas dataframe, Should contain a "yield" column
             
                    Usually obtained either from the raw data (perfect predictions)
                    or from a predictions dataframe
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    actions: Pandas dataframe, Should contain a "charge" column
             
             Actions are usually obtained from another rollout or optimization model. 
             Rollouts include: logic_rollout, actions_rollout, or pred_logic_rollout
             Optimization models include: DP and MPC (price, carb, both)
             
    
    Example: merged = merge("h16")
            
             rf = RF(house)
            
             pred = rf.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")
            
             actions = DP("2022-06-19 00:00:00","2022-06-19 23:00:00",pred,Battery(max_capacity=13),
                          byday=True,ints=True,degrade=False,verbose=False)
                
             series = action_rollout(merged.loc["2022-06-19 00:00:00":"2022-06-19 23:00:00"], Battery(max_capacity=13), actions)
    '''
    return _logic_rollout(series_battery, battery, actions)
    

def pred_logic_rollout(series_battery_true,series_battery_pred, battery):
    '''
    Function that applies the simple logic used in the report on predicted values,
    and then applies those actions to the raw data. If the battery has 
    max_capacity = 0, then this is also the "no battery" model
    
    Return type: Pandas dataframe
    
    Usage: When running the models on predictions; simple logic. Also is sanity check
           that "no battery" model does nothing still
    
    
    Input:
    
    series_battery_true: Pandas dataframe, the raw data (output of "merge" function)
             
                         This is the raw data the actions obtained from applying simple
                         logic to the predictions will be applied to. Should have the
                         same length as series_battery_pred.
    
    series_battery_pred: Pandas dataframe, the predictions (obtained either from
                         "rf" class "get_predictions" functions or "SARIMA" class
                         "SARIMA" function). Should contain a "yield" column
             
                         These are the predictions that are obtained from one of
                         the prediction models.
    
    battery: Battery instance, A class imported from "Battery.py"
             
             A simulated battery using the Battery class. For the report
             the battery has a max_capacity = 13.0 and a max_charge = 7.0.
             The logical actions will be applied to this battery.
             
    
    Example: merged = merge("h16")
            
             rf = RF(house)
            
             pred = rf.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")
             
             series = pred_logic_rollout(merged.loc["2022-06-19 00:00:00":"2022-06-19 23:00:00"], pred, Battery(max_capacity=13))
    '''
    
    series_battery_pred = logic_rollout(series_battery_pred, deepcopy(battery))
    
    series_battery_true = action_rollout(series_battery_true, battery, series_battery_pred)
    
    return series_battery_true
    
    
def print_price_summary(series_battery,yearprint=True):
    '''
    Prints information about series_battery, output from any rollout.
    Rollouts include: logic_rollout, actions_rollout, or pred_logic_rollout
    
    Prints period, cost of period, total emissions of period and number
    of kWh purchased and sold in period. If yearprint, it also prints cost per
    year, emissions per year, number of kWh purchased and sold per year.
    
    Return type: None
    
    Usage: Used when wanting to see information about the series, without looking
           at the whole series. If the whole series is wanted, see "logic_series_print"
           function
    
    
    Input:
    
    series_battery: Pandas dataframe, Should be output of rollout
             
                    This is either a logic or an optimization model run through
                    any rollout function.
             
    
    Example: print_price_summary(logic_rollout(merged.loc[Start:End], Battery(max_capacity=13)))
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
        print()
        print(f"Average cost per year is: {round(series_battery['cost'].sum()/years_timedelta,0)} DKK")
        print(f"Average emissions per year is: {round(series_battery['emission'].sum()/years_timedelta,0)} kg")
        
    print()
    print(f"Number of kwh purchased in the period: {-num_wh_total}")
    print(f"Number of kwh sold in the period: {num_wh_total_sold}")

    if yearprint:
        print()
        print(f"Average number of kwh purchased per year: {-num_wh_total/years_timedelta}")
        print(f"Average number of kwh sold per year: {num_wh_total_sold/years_timedelta}")

        
def logic_series_print(series_battery, p2p=False):
    '''
    Prints ALL values from series_battery, output from any rollout.
    Rollouts include: logic_rollout, actions_rollout, or pred_logic_rollout
    
    Return type: None
    
    Usage: Used to examine what actions were taken, prices, battery behaviour and 
           if the model resorted to buying more than necessary just to charge the
           battery
    
    
    Input:
    
    series_battery: Pandas dataframe, Should be output of rollout
             
                    This is either a logic or an optimization model run through
                    any rollout function.
                    
    p2p: bool, optional input that's False by default
    
         If true, will assume the logical series has been updated to reflect it's
         actions in a p2p setting: There's now a "grid" and "peer" column that 
         show how much it has bought from the grid and from its peers at that timestep.
             
    
    Example: logic_series_print(logic_rollout(merged.loc[Start:End], Battery(max_capacity=13)))
    '''
    if not p2p:
        print(f"{' hour':5s}  {'   price':8s}  {'  eprice':8s}  {' yield':6s} {' surpl':6s} {'  ext':5s} {'  act':5s} {'   bef':6s} {'   deg':6s} {'   aft':6s}  {'    cost':8s}  {' pcumsum':8s}  {'    emis':8s}  {' ecumsum':8s}")

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
        
            print(f"{i:5d}: {spot:8.4f}, {eprice:8.4f}, {yieldd:6.1f},{surplus:6.1f},{buy:5.1f},{charge:5.1f},{before:6.1f},{degrade:6.1f},{after:6.1f}, {cost:8.4f}, {cost_c:8.4f}, {emis:8.4f}, {emis_c:8.4f}")
            
    else:
        print(f"{' hour':5s}  {'   price':8s}  {'  eprice':8s}  {' yield':6s} {' surpl':6s} {' grid':5s} {' peer':5s} {'  ext':5s} {'  act':5s} {'   bef':6s} {'   deg':6s} {'   aft':6s}  {'    cost':8s}  {' pcumsum':8s}  {'    emis':8s}  {' ecumsum':8s}")

        for i in range(len(series_battery)):
            spot    = series_battery.iloc[i]['SpotPriceDKK']/1000
            eprice  = series_battery.iloc[i]['CO2Emission']/1000
            yieldd  = series_battery.iloc[i]['yield']
            surplus = series_battery.iloc[i]['surplus']
            grid    = series_battery.iloc[i]['grid']
            peer    = series_battery.iloc[i]['peer']
            buy     = series_battery.iloc[i]['buy']
            charge  = series_battery.iloc[i]['charge']
            before  = series_battery.iloc[i]['capacity_before']
            degrade = series_battery.iloc[i]['capacity_degraded']
            after   = series_battery.iloc[i]['capacity_after']
            cost    = series_battery.iloc[i]['cost']
            emis    = series_battery.iloc[i]['emission']
            cost_c  = series_battery.iloc[i]['cost_cummulative']
            emis_c  = series_battery.iloc[i]['emission_cummulative']
        
            print(f"{i:5d}: {spot:8.4f}, {eprice:8.4f}, {yieldd:6.1f},{surplus:6.1f},{grid:5.1f},{peer:5.1f},{buy:5.1f},{charge:5.1f},{before:6.1f},{degrade:6.1f},{after:6.1f}, {cost:8.4f}, {cost_c:8.4f}, {emis:8.4f}, {emis_c:8.4f}")
        

if __name__ == "__main__":
    print("This file is meant to be imported")
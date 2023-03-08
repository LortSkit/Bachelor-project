# Simple logic battery
def logic_bat(row, battery, actions=None): #actions only used for DP
    power_yield = row["power_yield"]
    battery.charge(power_yield)
    
    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = battery.get_surplus()
    row["charge"] = battery.charge_list[-1]
    row["buy"] = 0.0
    return row

# DP logic (mostly to compare to the other logic based models)
def logic_DP(row, battery, actions):
    power_yield = row["power_yield"]
    charge = actions.loc[row.name]["charge"]
    buy = actions.loc[row.name]["buy"]
    battery.charge(charge)
    
    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = power_yield-charge
    row["charge"] = charge
    row["buy"] = buy
    return row

if __name__ == "__main__":
    print("This file is meant to be imported")
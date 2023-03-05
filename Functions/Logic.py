# Simple logic battery
def logic_bat(row, battery):
    power_yield = row["power_yield"]
    battery.charge(power_yield,True)
    
    row["power_yield"] = power_yield
    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = battery.get_surplus()
    row["charge"] = battery.charge_list[-1]
    row["buy"] = 0.0
    return row

# DP logic (mostly to compare to the other logic based models)
def logic_DP(row, battery):
    power_yield = row["power_yield"]
    charge = battery.actions.loc[row.name]["charge"]
    buy = battery.actions.loc[row.name]["buy"]
    battery.charge(charge,True)
    
    row["power_yield"] = power_yield
    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = power_yield-charge
    row["charge"] = charge
    row["buy"] = buy
    return row

if __name__ == "__main__":
    print("This is a class meant to be imported")
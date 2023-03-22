# Simple logic battery
def logic_bat(row, battery, actions=None): #actions only used for DP
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
    yieldd = row["yield"]
    charge = actions.loc[row.name]["charge"]

    if yieldd<=0:
        buy=charge if charge>0 else 0.0
    else:
        if yieldd<charge:
            buy=charge-yieldd
        else:
            buy=0.0

    battery.charge(charge)

    row["capacity_before"] = battery.get_previous_capacity()
    row["capacity_degraded"] = battery.get_previous_degraded_capacity()
    row["capacity_after"] = battery.get_current_capacity()
    row["surplus"] = yieldd-charge
    row["charge"] = charge
    row["buy"] = buy
    return row

if __name__ == "__main__":
    print("This file is meant to be imported")
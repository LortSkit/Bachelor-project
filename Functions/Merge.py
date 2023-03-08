import pandas as pd

el = pd.read_csv('Elspotprices.csv', sep=';', decimal=',')
el = el.drop(['HourUTC','SpotPriceEUR'],axis=1)
el['HourDK'] = pd.to_datetime(el['HourDK'])

el_dk1 = el[el['PriceArea']=='DK1']

timestamps = pd.DataFrame()
timestamps['HourDK'] = pd.date_range(el['HourDK'].min(), el['HourDK'].max(),freq='H')

el_dk1=el_dk1.merge(timestamps,on='HourDK',how='right')

el_dk1 = el_dk1.ffill() 

el_dk1 = el_dk1.set_index('HourDK')
el_dk1 = el_dk1.drop(['PriceArea'],axis=1)

el_dk1 = el_dk1.loc["2020-12-22 00:00:00":"2023-01-09 20:00:00"] #Start at earliest prod/cons value, end before nans


pf_filled = pd.read_csv('pf_filled.csv')
cf_filled = pd.read_csv('cf_filled.csv')
df = pf_filled.merge(cf_filled)
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)

def merge(house=None):
    if house is None:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
    elif house.lower() not in ["k28","h16","h22","h28","h32"]:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
        
    production = df.loc[:"2023-01-09 20:00:00"]["prod_"+house]  #end before nans in spotprices
    consumption = df.loc[:"2023-01-09 20:00:00"]["cons_"+house] #end before nans in spotprices
    merged = pd.merge(production, consumption, how="outer", left_index=True,right_index=True)
    merged["power_yield"] = merged["prod_"+house] - merged["cons_"+house]
    merged = pd.merge(merged, el_dk1, how="outer", left_index=True,right_index=True)
    merged = merged
    return merged

if __name__ == "__main__":
    print("This file is meant to be imported")
import pandas as pd

#Spotprice data clean
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

el_dk1 = el_dk1.loc["2020-12-22 00:00:00":"2022-12-31 23:00:00"] #Start at earliest prod/cons value, end before nans

#Emissions data clean
em = pd.read_csv('carbon_emissions_data.csv', sep=',', decimal='.')
em = em.drop(['Unnamed: 0'], axis=1)
em['Minutes5DK'] = pd.to_datetime(em['Minutes5DK'])
em_dk1 = em[em['PriceArea']=='DK1']
em_dk1 = em_dk1.drop(['PriceArea'], axis=1)

timestamps = pd.DataFrame()
timestamps['Minutes5DK'] = pd.date_range(em['Minutes5DK'].min(), em['Minutes5DK'].max(),freq='5T')

em_dk1 = em_dk1.merge(timestamps,on='Minutes5DK',how='right')

em_dk1 = em_dk1.ffill() 

em_dk1 = em_dk1.set_index('Minutes5DK')

em_dk1 = em_dk1.asfreq('H').loc["2020-12-22 00:00:00":"2022-12-31 23:00:00"]

#Already cleaned production and consumption data
pf_filled = pd.read_csv('pf_filled.csv')
cf_filled = pd.read_csv('cf_filled.csv')
df = pf_filled.merge(cf_filled)
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)

#Make into one data frame (by house)
def merge(house=None):
    if house is None:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
    elif house.lower() not in ["k28","h16","h22","h28","h32"]:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
        
    production = df.loc[:"2022-12-31 23:00:00"]["prod_"+house]  
    consumption = df.loc[:"2022-12-31 23:00:00"]["cons_"+house] 
    merged = pd.merge(production, consumption, how="outer", left_index=True,right_index=True)
    merged["yield"] = merged["prod_"+house] - merged["cons_"+house]
    
    values = merged["yield"].to_numpy()
    
    for i in range(len(values)):
        values[i] = f"{values[i]:.1f}"
    
    merged["yield"] = values
    
    merged = pd.merge(merged, el_dk1, how="outer", left_index=True,right_index=True)
    merged = pd.merge(merged, em_dk1, how="outer", left_index=True,right_index=True)
    merged = merged
    return merged

if __name__ == "__main__":
    print("This file is meant to be imported")
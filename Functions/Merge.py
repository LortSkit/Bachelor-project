import pandas as pd

pf_filled = pd.read_csv('pf_filled.csv')
cf_filled = pd.read_csv('cf_filled.csv')
df = pf_filled.merge(cf_filled)
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)

el = pd.read_csv('Elspotprices.csv', sep=';', decimal=',')
el = el.drop(['HourUTC','SpotPriceEUR'],axis=1)
el['HourDK'] = pd.to_datetime(el['HourDK'])

#plt.rcParams["figure.figsize"] = (20,3)
#plt.plot(el[el['PriceArea']=='DK1']['SpotPriceDKK'],label='DK1',alpha=0.5)
#plt.plot(el[el['PriceArea']=='DK2']['SpotPriceDKK'],label='DK2',alpha=0.5)
#plt.legend()
#plt.show()

el_dk1 = el[el['PriceArea']=='DK1']

timestamps = pd.DataFrame()
timestamps['HourDK'] = pd.date_range(el['HourDK'].min(), el['HourDK'].max(),freq='H')

el_dk1=el_dk1.merge(timestamps,on='HourDK',how='right')

el_dk1 = el_dk1.ffill() ### Sønderborg is in dk1

el_dk1 = el_dk1.set_index('HourDK')
el_dk1 = el_dk1.drop(['PriceArea'],axis=1)

el_dk1 = el_dk1.iloc[3*365*24+3-9*24:]


pf_filled = pd.read_csv('pf_filled.csv')
cf_filled = pd.read_csv('cf_filled.csv')
df = pf_filled.merge(cf_filled)
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)

def merge(house=None,names=["prod","cons"]):
    if house is None:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
    elif house.lower() not in ["k28","h16","h22","h28","h32"]:
        raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
    
    if names is None or len(names)!=2:
        raise Exception("Second input must be a string list of len 2: names=['prod','cons']")
    production = df[names[0]+"_"+house]
    consumption = df[names[1]+"_"+house]
    merged = pd.merge(production, consumption, how="outer", left_index=True,right_index=True)
    merged["power_yield"] = merged[names[0]+"_"+house] - merged[names[1]+"_"+house]
    merged = pd.merge(merged, el_dk1, how="outer", left_index=True,right_index=True)
    merged = merged[:-24] #missing last day because no spot prices for the last day
    return merged

if __name__ == "__main__":
    print("This is a class meant to be imported")
import pandas as pd
from Merge import merge

def load_series(datatype=None,house=None,rename=None, shifts=None):
    '''
    Dataframe of either production, consumption or emissions values, along with its
    lagged features (shifts), as well as the features year, month, day, weekday, and
    hour, for the house specified.
    
    Return type: Pandas dataframe
    
    Usage: Feature matrix for when predicting using Random Forest, plotting purposes
    
    
    Input:
    
    datatype: str, specifies whether the production or consumption values are selected
              
              "Production", "Consumption", "Emissions", "P", "C", "E", "prod", "cons", 
              "emis" etc.
    
    house: str, specifies which house is selected (Required if datatype is not Emissions)
           
           "k28", "h16", "h22", "h28", or "h32"
         
    rename: str, renames chosen column. (Optional)
            
            "val" will rename the data type string and any shift names will start with 
            "val". If None, shift names will start with 't'.
            
    shifts: list of int, if given, specifies what shifted columns to add
            
            For report, shifts = [24, 48, 168]
           
           
    Example: train_series = load_series("p","h16","val", [24,48,168])
    '''
    res = None
    #Production or consumption string stuff
    datatypeStr = ""
    if datatype is None:
        raise Exception("First input must specify either Production or Consumption, or Emissions")
    elif datatype[0].lower()=="p":
        datatypeStr = "prod_"
    elif datatype[0].lower()=="c":
        datatypeStr = "cons_"
    elif datatype[0].lower()=="e":
        datatypeStr = "carb_"
    
    #Making sure an existing house is chosen
    if house is None and datatypeStr != "carb_":
        raise Exception("Second input must specify house: k28, h16, h22, h28, or h32")
    elif datatypeStr != "carb_" and house.lower() not in ["k28","h16","h22","h28","h32"]:
        raise Exception("Second input must specify house: k28, h16, h22, h28, or h32")
    
    #Production, consumption or emissions is chosen and loaded
    if datatypeStr=="prod_":
        res = pd.read_csv("pf_filled.csv",sep=",")
        res = res[["Time",datatypeStr+house]]
        res["Time"] = pd.to_datetime(res["Time"], utc=True)
        res = res.set_index("Time").sort_index()
        res.index = pd.date_range(start=res.index.tz_convert(None)[0], end=res.index.tz_convert(None)[-1], freq="h")
        res = res.loc["2020-12-22 00:00:00":"2022-12-31 23:00:00"]
    elif datatypeStr=="cons_":
        res = pd.read_csv("cf_filled.csv",sep=",")
        res = res[["Time",datatypeStr+house]]
        res["Time"] = pd.to_datetime(res["Time"], utc=True)
        res = res.set_index("Time").sort_index()
        res.index = pd.date_range(start=res.index.tz_convert(None)[0], end=res.index.tz_convert(None)[-1], freq="h")
        res = res.loc["2020-12-22 00:00:00":"2022-12-31 23:00:00"]
    else:
        merged=merge("h16").loc["2020-12-22 00:00:00":"2022-12-31 23:00:00"] #placeholder house: use merge to get emissions
        res = pd.DataFrame()
        if rename is not None:
            res[rename] = merged["CO2Emission"]
            carb_name = rename
        else:
            res["CO2Emission"] = merged["CO2Emission"] 
            carb_name = "CO2Emission"
        res["Time"] = merged.index
        res = res[["Time",carb_name]]
        res = res.set_index("Time").sort_index()
        res.index = pd.date_range(start=res.index[0], end=res.index[-1], freq="h")
    
    #Renaming value column
    if not house is None and not rename is None and datatypeStr!="carb_":
        res = res.rename(columns={datatypeStr+house: rename}, errors="raise")
    
    #Shifted columns added: lagged features
    if not shifts is None:
        if rename is None:
            shiftname = 't'
        else:
            shiftname = rename
        dataframe = pd.DataFrame()
        for i in shifts:
            dataframe[shiftname + '-' + str(i)] = res.shift(i)
        res = pd.concat([res, dataframe], axis=1)
        res.dropna(inplace=True)
    
    #Time features
    res['Year']=res.index.year
    res['Month']=res.index.month
    res['Day']=res.index.day
    res['Weekday']=res.index.weekday
    res['Hour']=res.index.hour
    return res

def moving_average(timeseries, window):
    '''
    Used to smoothen timeseries for plotting purposes
    
    Return type: Pandas series
    
    Usage: Plotting timeseries
    
    
    Input:
    
    timeseries: Pandas series, series to be plotted
    
                Is usually series like production or consumption
                
    window: int, how many data points to consider the mean (average) over
    
            When wanting to see all the the data, 3 is usually used. Depending on
            what you want to show, you should change this to something that looks
            nice.
    
    
    Example: import matplotlib.pyplot as plt
    
             fig, ax=plt.subplots(figsize=(11, 3))
    
             moving_average(df['val'], 3).plot(ax=ax, label='val', color='blue')
             
             ax.legend();
             plt.show()
    '''
    return timeseries.rolling(window=window, center=True).mean()

if __name__ == "__main__":
    print("This file is meant to be imported")
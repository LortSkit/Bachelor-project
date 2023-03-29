import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from Merge import merge

class SARIMA:
    def __init__(self, house, cons_ts = 500, prod_ts=3000, carb_ts=500):
        self.house = house
        self.cons_ts = cons_ts
        self.prod_ts = prod_ts
        self.carb_ts = carb_ts
    
        self.df = merge(self.house)
    
    def model(self, start_time, end_time, var):
        n = len(pd.date_range(start_time, end_time,freq='H'))
        
        if var=='prod':
            train_size = self.prod_ts
            string = var+'_'+str(self.house)
        elif var == 'cons':
            train_size = self.cons_ts
            string = var+'_'+str(self.house)
        else:
            train_size = self.carb_ts
            string = 'CO2Emission'
        
        train_ind = pd.date_range(end=start_time, periods=train_size+1, freq="H")
        train = self.df[string].loc[train_ind[0]:train_ind[-2]]
        test = self.df[string].loc[start_time:end_time]
        
        arima_model = auto_arima(train, start_p=0, start_d=0, start_q=0, 
                        max_p = 5, max_d=5, max_q=5, start_P=0,
                        start_D=0, start_Q=0, m=24, seasonal=True, 
                        error_action='warn', trace=True, supress_warning=True,
                        stepwise=True, random_state=20, n_fits=50)
        
        pred = arima_model.predict(n_periods=n)    
        rmse = np.sqrt(mean_squared_error(test,pred))
        
        tf = pd.DataFrame(columns=['test','pred'])
        tf['test'] = test
        tf['pred'] = pred
        return tf
    
    def SARIMA(self, start_time, end_time):
        pf = self.model(start_time, end_time, 'prod')
        cf = self.model(start_time, end_time, 'cons')
        carbf = self.model(start_time, end_time, 'carbon')
        price = self.df['SpotPriceDKK'].loc[start_time:end_time]
        
        of = pd.DataFrame()
        of['prod_'+str(self.house)] = pf['pred'].to_numpy()
        of['cons_'+str(self.house)] = cf['pred'].to_numpy()
        of['yield'] = of['prod_'+str(self.house)] - of['cons_'+str(self.house)]
        of['SpotPriceDKK'] = (self.df['SpotPriceDKK'].loc[start_time:end_time]).to_numpy()
        of['CO2Emission'] = carbf['pred'].to_numpy()
        
        of.set_index(df.loc[start_time:end_time].index,inplace=True)
        return of

if __name__ == "__main__":
    print("This file is meant to be imported")
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import bayesian_search_forecaster

from skopt.space import Categorical, Real, Integer
import multiprocessing

from Merge import merge
from LoadSeries import load_series

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import time
    
#Globals for class definition
prod = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123,max_depth=12,max_features='log2',n_estimators=412,n_jobs=4),
                 lags      = list(range(1,24+1,1))
                )
cons = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123,max_depth=9,max_features='log2',n_estimators=633,n_jobs=4),
                 lags      = list(range(1,12+1,1))
                )

carb = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123,max_depth=12,max_features='log2',n_estimators=412,n_jobs=4),
                 lags      = list(range(1,24+1,1))
                )

merged = merge("h16") #Used for getting spotprices

class RF:
    def __init__(self, house, forecasterProd=prod, forecasterCons=cons, forecasterCarb=carb, prod_ts = 4000, cons_ts=500, carb_ts=1000):
        if house is None:
            raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
        elif house.lower() not in ["k28","h16","h22","h28","h32"]:
            raise Exception("First input must specify house: k28, h16, h22, h28, or h32")
        
        self.house = house
        self.forecasterProd = forecasterProd
        self.forecasterCons = forecasterCons
        self.forecasterCarb = forecasterCarb
        self.prod_ts = prod_ts
        self.cons_ts = cons_ts
        self.carb_ts = carb_ts
        self.df_prod = load_series("p",house,"prod_"+house,[24,48,168])
        self.df_cons = load_series("c",house,"cons_"+house,[24,48,168])
        self.df_carb = load_series("e",house,"carb_"+house,[24,48,168])
      
    def get_predictions(self, Start="2022-06-19 00:00:00", End="2022-06-19 23:00:00"):
        BeforeProd = pd.date_range(end=Start, periods=self.prod_ts+1, freq="h")[0]
        BeforeCons = pd.date_range(end=Start, periods=self.cons_ts+1, freq="h")[0]
        BeforeCarb = pd.date_range(end=Start, periods=self.carb_ts+1, freq="h")[0]
        
        _, predProd = backtesting_forecaster(forecaster = self.forecasterProd,
                                             y          = self.df_prod.loc[BeforeProd:End]['prod_'+self.house],
                                             exog       = self.df_prod.loc[BeforeProd:End].drop(['prod_'+self.house], axis=1),
                                             initial_train_size = self.prod_ts,
                                             steps      = 24,
                                             refit      = True,
                                             metric     = 'mean_squared_error',
                                             fixed_train_size   = True,
                                             verbose    = False
                                            )

        _, predCons = backtesting_forecaster(forecaster = self.forecasterCons,
                                             y          = self.df_cons.loc[BeforeCons:End]['cons_'+self.house],
                                             exog       = self.df_cons.loc[BeforeCons:End].drop(['cons_'+self.house], axis=1),
                                             initial_train_size = self.cons_ts,
                                             steps      = 24,
                                             refit      = True,
                                             metric     = 'mean_squared_error',
                                             fixed_train_size   = True,
                                             verbose    = False
                                            )

        _, predCarb = backtesting_forecaster(forecaster = self.forecasterCarb,
                                             y          = self.df_carb.loc[BeforeCarb:End]['carb_'+self.house],
                                             exog       = self.df_carb.loc[BeforeCarb:End].drop(['carb_'+self.house], axis=1),
                                             initial_train_size = self.carb_ts,
                                             steps      = 24,
                                             refit      = True,
                                             metric     = 'mean_squared_error',
                                             fixed_train_size   = True,
                                             verbose    = False
                                            )

        pred = pd.merge(predProd, predCons, how="outer", left_index=True,right_index=True)
        pred = pd.merge(pred, predCarb, how="outer", left_index=True,right_index=True)
        pred = pred.rename(columns={"pred_x":"prod_"+self.house,"pred_y":"cons_"+self.house, "pred":"CO2Emission"})

        pred["prod_"+self.house]=np.array((pred["prod_"+self.house].to_numpy())*10,dtype=int)/10
        pred["cons_"+self.house]=np.array((pred["cons_"+self.house].to_numpy())*10,dtype=int)/10
        pred["CO2Emission"]=np.array((pred["CO2Emission"].to_numpy())*10,dtype=int)/10

        pred["yield"] = pred["prod_"+self.house] - pred["cons_"+self.house]

        values = pred["yield"].to_numpy()

        for j in range(len(values)):
            values[j] = f"{values[j]:.1f}"

        pred["yield"] = values
        
        
        pred["SpotPriceDKK"] = self.get_spotprices(Start, End)

        return pred
                
    def tune_hyperparameters(self, datatype, Start="2021-06-19 00:00:00", End="2021-06-19 23:00:00", stepsize=2, train_sizes=[100,500,1000,2000,3000,4000], lags_grid=[1,6,12,18,24], n_trials = 10):   
        if datatype.lower()[0] not in ["p","c","e"]:
            raise Exception("First input must specify either Production or Consumption, or Emissions")
        elif datatype[0].lower()=="p":
            train_series = self.df_prod
        elif datatype[0].lower()=="c":
            train_series = self.df_cons
        elif datatype[0].lower()=="e":
            train_series = self.df_carb
        

        search_space = {'n_estimators' : Integer(2, 1000, "uniform", name='n_estimators'),
                        'max_depth'    : Integer(1, 20, "uniform", name='max_depth'),
                        'max_features' : Categorical(['log2', 'sqrt'], name='max_features')}

        Befores = [pd.date_range(end=Start, periods=train_size+1, freq="h")[0] for train_size in train_sizes]
        
        print(f"Tuning on period from {Start} to {End}")

        start = time.time()
        results_i = []
        metrics   = []
        for i in range(len(train_sizes)):
            #Initialize
            forecaster = ForecasterAutoreg(
                            regressor = RandomForestRegressor(random_state=123),
                            lags      = 1 # This value will be replaced in the grid search
                         )

            train_size = train_sizes[i]
            Before     = Befores[i]

            #Tune
            results, frozen_trial = bayesian_search_forecaster(
                                        forecaster         = forecaster,
                                        y                  = train_series.loc[Before:End]['val'],
                                        exog               = train_series.loc[Before:End].drop(['val'], axis=1),
                                        lags_grid          = lags_grid,
                                        search_space       = search_space,
                                        steps              = 24,
                                        metric             = 'mean_squared_error',
                                        refit              = True,
                                        initial_train_size = train_size,
                                        fixed_train_size   = True,
                                        n_trials           = n_trials,
                                        random_state       = 123,
                                        return_best        = True,
                                        verbose            = False,
                                        engine             = 'skopt',
                                        kwargs_gp_minimize = {})
            end = time.time()
            print("Time spent so far {}s".format(end-start))
            metric = results.mean_squared_error.to_numpy()[0]
            argmin = results.mean_squared_error.index.to_numpy()[0]
            metrics.append(metric)
            results_i.append(results.loc[argmin])

        i=np.argmin(metrics)
        best = results_i[i]
        train_size = train_sizes[i]
        
        print(f"Best training size = {train_size}")
        print(f"Best lags = {best.lags}")
        print(f"Best parameters = {best.params}")
        
        forecaster = self.get_forecaster(train_size, best.lags, best.params)
        
        self.set_forecaster(datatype,forecaster)
    
    def set_forecaster(self, datatype, forecaster):
        if datatype.lower()[0] not in ["p","c","e"]:
            raise Exception("First input must specify either Production or Consumption, or Emissions")
        elif datatype[0].lower()=="p":
            self.forecasterProd = forecaster
        elif datatype[0].lower()=="c":
            self.forecasterCons = forecaster
        elif datatype[0].lower()=="e":
            self.forecasterCarb = forecaster
    
    def get_forecaster(self, ts, lags, params):
        forecaster = ForecasterAutoreg(
                         regressor = RandomForestRegressor(random_state=123,**params),
                         lags      = lags)
        
        return forecaster
            
    def get_spotprices(self, Start, End):
        return merged.loc[Start:End]["SpotPriceDKK"]

if __name__ == "__main__":
    print("This file is meant to be imported")
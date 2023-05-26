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

merged = merge("h16") #Used for getting spotprices and emissions

class RF:
    '''
    RF class, made to get predictions and to hyperparameter tune
    
    When getting predictions, the internal forecasters and training sizes
    are used, which are either set upon initialization or set either manually
    or automatically when hyperparameter tuning.
    
    Usage: This sees use whenever predictions are needed for this project,
           which is either during single house or Peer-to-Peer optimization
    
    
    Input:
    
    house: str, specifies which house is selected
           
           "k28", "h16", "h22", "h28", or "h32"
           
    forecasterProd: ForecasterAutoreg instance, The production forecaster
    
                    This input is optional, as you may want to hyperparamter tune
                    using this instance to get the forecaster
                    
    forecasterCons: ForecasterAutoreg instance, The consumption forecaster
    
                    This input is optional, as you may want to hyperparamter tune
                    using this instance to get the forecaster
                    
    forecasterCarb: ForecasterAutoreg instance, The carbon emissions forecaster
    
                    This input is optional, as you may want to hyperparamter tune
                    using this instance to get the forecaster
                    
    prod_ts: int, The training size of the production forecaster
    
             This input is optional, as you may want to hyperparamter tune
             using this instance to get the training size
             
    cons_ts: int, The training size of the consumption forecaster
    
             This input is optional, as you may want to hyperparamter tune
             using this instance to get the training size
             
    carb_ts: int, The training size of the carbon emissions forecaster
    
             This input is optional, as you may want to hyperparamter tune
             using this instance to get the training size
     
     
    Example: rf_model = RF(self.house)
        
             rf_model.tune_hyperparameters("p","2021-06-19 00:00:00", "2021-06-19 23:00:00")
             rf_model.tune_hyperparameters("c","2021-06-19 00:00:00", "2021-06-19 23:00:00")
             rf_model.tune_hyperparameters("e","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 
             pred = rf_model.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00") 
    
    
    Additional attributes:
    
    df_prod: Pandas dataframe, production feature matrix
    
             Used for predicting production values
             
    df_cons: Pandas dataframe, consumption feature matrix
    
             Used for predicting consumption values
      
    df_carb: Pandas dataframe, carbon emissions feature matrix
    
             Used for predicting emission values
    
     
    Functions (check their docs):
    
    get_predictions: (str or Pandas Timestamp) * (str or Pandas Timestamp) * bool -> Pandas dataframe
    
                                                  or
                                                  
                     (str or Pandas Timestamp) * (str or Pandas Timestamp) -> Pandas dataframe                           
    
    tune_hyperparameters: (str or Pandas Timestamp) * (str or Pandas Timestamp) * int list * int list * int -> None
    
                                                  or
                                                                  
                          (str or Pandas Timestamp) * (str or Pandas Timestamp) -> None
    
    set_forecaster: str * int * ForecasterAutoreg instance -> None
    
    def get_forecaster(self, lags, params)
    
    get_forecaster: int list * RandomForestRegressor params dict -> ForecasterAutoreg instance
    
    get_spotprices: (str or Pandas Timestamp) * (str or Pandas Timestamp) -> Pandas series
    
    get_emissions: (str or Pandas Timestamp) * (str or Pandas Timestamp) -> Pandas series
    '''
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
      
    def get_predictions(self, Start="2022-06-19 00:00:00", End="2022-06-19 23:00:00", get_carb = True):
        '''
        Returns a predictions dataframe ready for use in the optimization models
        
        When getting predictions, the internal forecasters and training sizes
        are used, which are either set upon initialization or set either manually
        or automatically when hyperparameter tuning.

        Return type: Pandas dataframe

        Usage: This sees use whenever predictions are needed for this project,
               which is either during single house or Peer-to-Peer optimization


        Input:
    
        Start: str or Pandas Timestamp, Defines the starting date of the model
            
               Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
               If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
        End: str or Pandas Timestamp, Defines the end date of the model

             Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
             If timestamp: pd.Timestamp("2020-12-22 00:00:00")
             
        get_carb: bool, optional, default set to True
                  
                  Specifies whether carbon should be predicted.
                  If False, it uses the true values. This is useful incase of
                  Peer-to-Peer optimization, as only one house would need to
                  predict carbon, as it would be the same for all houses.


        Example: rf_model = RF(self.house)
        
                 rf_model.tune_hyperparameters("p","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 rf_model.tune_hyperparameters("c","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 rf_model.tune_hyperparameters("e","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 
                 pred = rf_model.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00")  
        '''
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

        if get_carb:
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
        else:
            predCarb = self.get_emissions(Start, End)

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
                
    def tune_hyperparameters(self, datatype, Start="2021-06-19 00:00:00", End="2021-06-19 23:00:00", train_sizes=[100,500,1000,2000,3000,4000], lags_grid=[1,6,12,18,24], n_trials = 10): 
        '''
        Hyperparameters tunes for given period. Should leave enough hours before
        Start date for the greates train_size in train_sizes. Earliest date in
        dataset is "2020-12-22 00:00:00".
        
        Saves the best found hyperparameters and train_size to the specified forecaster

        Return type: None

        Usage: Before getting predictions for any period, production, consumption
               and carbon emissions forecasters should be hyperparameter tuned.


        Input:
        
        datatype: str, should start with either p, c, or e
    
                  If first letter is "p", tunes production forecaster
                  If first letter is "c", tunes consumption forecaster
                  If first letter is "e", tunes carbon emissions forecaster
    
        Start: str or Pandas Timestamp, Defines the starting date of the model
            
               Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
               If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
        End: str or Pandas Timestamp, Defines the end date of the model

             Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
             If timestamp: pd.Timestamp("2020-12-22 00:00:00")
                  
        train_sizes: int list, optional, specifies which train sizes to inspect
        
                     For the project, we use 
                     train_sizes = [100,500,1000,2000,3000,4000], which is the
                     default value.
                  
        lags_grid: int list, specifies which lags sizes to inspect
        
                   For the project, we use lags_grid = [1,6,12,18,24], 
                   which is the default value.
                  
        n_trials: int, Number of parameter settings that are sampled in each lag
        
                  Used by skforecast's bayesian_search_forecaster


        Example: rf_model = RF(self.house)
        
                 rf_model.tune_hyperparameters("p","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 rf_model.tune_hyperparameters("c","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 rf_model.tune_hyperparameters("e","2021-06-19 00:00:00", "2021-06-19 23:00:00")
                 
                 pred = rf_model.get_predictions("2022-06-19 00:00:00", "2022-06-19 23:00:00") 
        '''
        if datatype.lower()[0] not in ["p","c","e"]:
            raise Exception("First input must specify either Production or Consumption, or Emissions")
        elif datatype[0].lower()=="p":
            train_series = self.df_prod
            datatype_str = "prod_"+self.house
        elif datatype[0].lower()=="c":
            train_series = self.df_cons
            datatype_str = "cons_"+self.house
        elif datatype[0].lower()=="e":
            train_series = self.df_carb
            datatype_str = "carb_"+self.house
        

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
                                        y                  = train_series.loc[Before:End][datatype_str],
                                        exog               = train_series.loc[Before:End].drop([datatype_str], axis=1),
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
        
        forecaster = self.get_forecaster(best.lags, best.params)
        
        self.set_forecaster(datatype,train_size,forecaster)
    
    def set_forecaster(self, datatype, ts, forecaster):
        '''
        Sets the train size and forecaster for specified datatype
        to be inputted values.

        Return type: None

        Usage: Used in tune_hyperparameters


        Input:
    
        datatype: str, should start with either p, c, or e
    
                  If first letter is "p", tunes production forecaster
                  If first letter is "c", tunes consumption forecaster
                  If first letter is "e", tunes carbon emissions forecaster
                
        ts: int, Training size the forecaster needs
                                
            Obtained from hyperparameter tuning
                
        forecaster: ForecasterAutoreg instance, forecaster
                                
                    Obtained from output of get_forecaster


        Example: (Used in tune_hyperparameters)
        '''
        if datatype.lower()[0] not in ["p","c","e"]:
            raise Exception("First input must specify either Production or Consumption, or Emissions")
        elif datatype[0].lower()=="p":
            self.forecasterProd = forecaster
            self.prod_ts = ts
        elif datatype[0].lower()=="c":
            self.forecasterCons = forecaster
            self.cons_ts = ts
        elif datatype[0].lower()=="e":
            self.forecasterCarb = forecaster
            self.carb_ts = ts
    
    def get_forecaster(self, lags, params):
        '''
        Creates ForecasterAutoreg instance forecaster from input
        lags and parameters

        Return type: ForecasterAutoreg instance

        Usage: Used in tune_hyperparameters


        Input:
    
        lags: int list, specifies the chosen lags
        
              Part of output of bayesian_search_forecaster, which
              is used in tune_hyperparameters
                
        params: RandomForestRegressor params dict, specifies chosen hyperparameters
                                
                Part of output of bayesian_search_forecaster, which
                is used in tune_hyperparameters


        Example: (Used in tune_hyperparameters)
        '''
        forecaster = ForecasterAutoreg(
                         regressor = RandomForestRegressor(random_state=123,**params),
                         lags      = lags)
        
        return forecaster
            
    def get_spotprices(self, Start, End):
        '''
        Returns the spotprices for specified period

        Return type: Pandas series

        Usage: Used in get_predictions


        Input:
    
        Start: str or Pandas Timestamp, Defines the starting date of the model
            
               Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
               If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
        End: str or Pandas Timestamp, Defines the end date of the model

             Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
             If timestamp: pd.Timestamp("2020-12-22 00:00:00")


        Example: (Used in get_predictions)
        '''
        return merged.loc[Start:End]["SpotPriceDKK"]
    
    def get_emissions(self, Start, End):
        '''
        Returns the true emission values for specified period

        Return type: Pandas series

        Usage: Used in get_predictions ONLY when get_carb=False


        Input:
    
        Start: str or Pandas Timestamp, Defines the starting date of the model
            
               Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
               If timestamp: pd.Timestamp("2020-12-22 00:00:00")
           
        End: str or Pandas Timestamp, Defines the end date of the model

             Can be dates between "2020-12-22 00:00:00" and "2022-12-31 23:00:00"
             If timestamp: pd.Timestamp("2020-12-22 00:00:00")


        Example: (Used in get_predictions)
        '''
        return merged.loc[Start:End]["CO2Emission"]

if __name__ == "__main__":
    print("This file is meant to be imported")
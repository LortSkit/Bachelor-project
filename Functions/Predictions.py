import numpy as np
import pandas as pd
from skforecast.model_selection import backtesting_forecaster

def RF_Predictions(house,forecasterProd,forecasterCons,forecasterCarb,prod,cons,carb,train_prod,train_cons,train_carb,
                   BeforeProd,BeforeCons,BeforeCarb,End,spotprices):
    _, predProd = backtesting_forecaster(forecaster = forecasterProd,
                                         y          = prod.loc[BeforeProd:End]['prod_'+house],
                                         exog       = prod.loc[BeforeProd:End].drop(['prod_'+house], axis=1),
                                         initial_train_size = train_prod,
                                         steps      = 24,
                                         refit      = True,
                                         metric     = 'mean_squared_error',
                                         fixed_train_size   = True,
                                         verbose    = False
                                        )

    _, predCons = backtesting_forecaster(forecaster = forecasterCons,
                                         y          = cons.loc[BeforeCons:End]['cons_'+house],
                                         exog       = cons.loc[BeforeCons:End].drop(['cons_'+house], axis=1),
                                         initial_train_size = train_cons,
                                         steps      = 24,
                                         refit      = True,
                                         metric     = 'mean_squared_error',
                                         fixed_train_size   = True,
                                         verbose    = False
                                        )
    
    _, predCarb = backtesting_forecaster(forecaster = forecasterCarb,
                                         y          = carb.loc[BeforeCarb:End]['carb_'+house],
                                         exog       = carb.loc[BeforeCarb:End].drop(['carb_'+house], axis=1),
                                         initial_train_size = train_carb,
                                         steps      = 24,
                                         refit      = True,
                                         metric     = 'mean_squared_error',
                                         fixed_train_size   = True,
                                         verbose    = False
                                        )
    
    pred = pd.merge(predProd, predCons, how="outer", left_index=True,right_index=True)
    pred = pd.merge(pred, predCarb, how="outer", left_index=True,right_index=True)
    pred = pred.rename(columns={"pred_x":"prod_"+house,"pred_y":"cons_"+house, "pred":"CO2Emission"})

    pred["prod_"+house]=np.array((pred["prod_"+house].to_numpy())*10,dtype=int)/10
    pred["cons_"+house]=np.array((pred["cons_"+house].to_numpy())*10,dtype=int)/10
    pred["CO2Emission"]=np.array((pred["CO2Emission"].to_numpy())*10,dtype=int)/10
    
    pred["yield"] = pred["prod_"+house] - pred["cons_"+house]
    
    values = pred["yield"].to_numpy()
    
    for j in range(len(values)):
        values[j] = f"{values[j]:.1f}"
    
    pred["yield"] = values
    
    
    pred["SpotPriceDKK"] = spotprices
    
    return pred

if __name__ == "__main__":
    print("This file is meant to be imported")
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
import multiprocessing as mp

def generate_forecast(vendor_name, geo_region_name, geo_region_group):
    """
    A function to generate a forecast for a given vendor and geo region, and return the merged data with the forecast. 

    Parameters:
    vendor_name (str): The name of the vendor.
    geo_region_name (str): The name of the geographical region.
    geo_region_group (pd.DataFrame): The geographical region group data.

    Returns:
    pd.DataFrame: The merged data with the forecast for the given vendor and geo region.
    """
    print(f"Forecasting for Vendor: {vendor_name} in Geo-region:{geo_region_name}: Started")
    geo_region_group = geo_region_group.set_index('Invoice Date')
    geo_region_group = geo_region_group.sort_index()
    start_date = geo_region_group.index.min()
    end_date = geo_region_group.index.max()
    date_table = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['Date'])
    geo_region_group = geo_region_group.reset_index()
    merged_df = pd.merge(date_table, geo_region_group, left_on='Date', right_on='Invoice Date', how='outer')
    merged_df['Invoice total by day'] = merged_df['Invoice total by day'].fillna(0)
    merged_df.set_index('Date', inplace=True)
    
    if(len(geo_region_group)<50):

         # create log file .txt
        with open('log.txt', 'a') as f:
            f.write(f"Not enough data to forecast for Vendor: {vendor_name} in Geo-region:{geo_region_name}\n")
        print(f"Forecasting for Vendor: {vendor_name} in Geo-region:{geo_region_name}: Ended")

    else:
        # ignore the warning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Decompose the time series
        period_hypothesized = round(len(merged_df)/3)
        model = ExponentialSmoothing(merged_df['Invoice total by day'], trend='add', seasonal='add', seasonal_periods=period_hypothesized, freq='D')
        model_fit = model.fit()

        # Forecast the next periods
        forecast = model_fit.forecast(steps=300)
        forecast[forecast <= 0] = 0.00
        # forecast.plot()
        # plt.show()
        
        # merge the forecast with the original data
        forecast_df = pd.DataFrame(forecast, columns=['Forecasted Value'])
        merged_df = pd.concat([merged_df, forecast_df], axis=1)
        merged_df = merged_df.reset_index(drop=False)
        merged_df['Vendor'] = vendor_name
        merged_df['Geo Region Group'] = geo_region_name
        merged_df['Vendor ID'] = geo_region_group['Vendor ID'].iloc[0]
        merged_df.drop(columns=['Invoice Date'], inplace=True)
        merged_df['Invoice total by day'] = merged_df['Invoice total by day'].fillna(merged_df['Forecasted Value'])
        merged_df.drop(columns=['Forecasted Value'], inplace=True)

        # create log file .txt
        with open('log.txt', 'a') as f:
            f.write(f"Forecasted for Vendor: {vendor_name} in Geo-region:{geo_region_name}\n")
        
        print(f"Forecasting for Vendor: {vendor_name} in Geo-region:{geo_region_name}: Ended")
        return(merged_df)    
       

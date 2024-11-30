import pandas as pd
import numpy as np
import scipy


def load_data(start_date, end_date, selected_tickers=None, reduced=False, k=100):
    """
    Load data from files into 3 data tables. Picks data between relevant time periods and removes cols with NA. Returns
    tickers of included assets as well

    :param start_date: 'YYYY-MM-DD' string of start date
    :param end_date: 'YYYY-MM-DD' string of end date
    :param selected_tickers: list of tickers to manually select. By default will select all non NA cols
    :return: (price_data, share_data, asset_data) for relevant time period
    """
    # Load files with price and s/o data, and asset classes
    if reduced:
        t_prices = pd.read_csv('res/sorted_prices.csv')
        t_shares_outstanding = pd.read_csv('res/sorted_shares.csv')
        t_asset_classes = pd.read_csv('res/securities_type.csv')

        prices = t_prices.iloc[:,0:k].copy()
        shares_outstanding = t_shares_outstanding.iloc[:, 0:k].copy()
        asset_classes = t_asset_classes.iloc[0:k, :].copy()

    else:
        prices = pd.read_csv('res/sorted_prices.csv')
        shares_outstanding = pd.read_csv('res/sorted_shares.csv')
        asset_classes = pd.read_csv('res/securities_type.csv')

    # Set date as index and remove date column
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.index = prices['Date']
    prices = prices.drop(['Date'], axis=1)

    # Set date as index and remove date column
    shares_outstanding['Date'] = pd.to_datetime(shares_outstanding['Date'])
    shares_outstanding.index = shares_outstanding['Date']
    shares_outstanding = shares_outstanding.drop(['Date'], axis=1)

    # Set ticker as index and remove ticker column
    asset_classes.index = asset_classes['Ticker']
    asset_classes = asset_classes.drop(['Ticker'], axis=1)

    # Select only relevant entries by dates selected
    relevant_prices = prices[start_date: end_date]
    relevant_so = shares_outstanding[start_date: end_date]

    # Remove columns with NAs
    prices_selected = relevant_prices.dropna(axis=1)
    shares_selected = relevant_so.dropna(axis=1)

    # Manually override which stocks to select
    if selected_tickers is not None:
        prices_selected = prices_selected[selected_tickers]
        shares_selected = shares_selected[selected_tickers]

    # Get all selected assets
    tickers = prices_selected.columns.tolist()

    # Filter asset class table for only selected assets
    selected_asset_classes = asset_classes[asset_classes.index.isin(tickers)]

    return prices_selected, shares_selected, selected_asset_classes


def get_period_data(data, start_date, end_date):
    """
    Given a data for the entire training and rebalacing period, return slice for relevant period

    :param data: tuple (price_data, share_data, asset_data) returned from load_data()
    :param start_date: 'YYYY-MM-DD' string of start date
    :param end_date: 'YYYY-MM-DD' string of end date
    :return: tuple (price_data, share_data, asset_data) for relevant time period
    """
    price_data, share_data, asset_data = data

    relevant_prices = price_data[start_date: end_date]
    relevant_so = share_data[start_date: end_date]

    return relevant_prices, relevant_so, asset_data


def get_num_assets(df, key):
    """
    Returns the number of assets for any asset class in the dataframe
    :param df: dataframe containing list of tickers and their type
    :param key: The asset class to sum over
    :return: number of assets (int)
    """
    return (df.Type.values == key).sum()


def get_returns(price_df):
    """
    Get the log returns of the price data

    :param price_df: dataframe of prices for assets
    :return: data frame of returns for assets
    """
    returns = np.log(price_df) - np.log(price_df.shift(1))
    returns = returns.drop(index=price_df.index[0], axis=0)
    return returns


def get_asset_count(asset_df):
    """
    Get the number of assets of each asset class in the backtesting period. Return a dictionary of asset counts

    :param asset_df: dataframe containing list of tickers and asset class
    :return: dictionary of asset classes and number of assets
    """
    asset_classes = ['Alternatives', 'Bonds', 'Commodities', 'Equity ETF', 'REIT', 'Stock']

    # Set asset count to 0
    asset_count = {
        'alternatives': 0,
        'bonds': 0,
        'commodities': 0,
        'equity_etfs': 0,
        'reit': 0,
        'individual_stocks': 0
    }

    # get the asset count keys to update dictionary
    asset_count_ids = list(asset_count.keys())

    for i in range(len(asset_classes)):

        # Get number of assets of each class
        num_assets = get_num_assets(asset_df, asset_classes[i])

        # Update dictionary with this value
        asset_count[asset_count_ids[i]] = num_assets

    return asset_count


def construct_allocation_matrix(asset_count):
    """
    Construct the C matrix from asset class dictionary

    :param asset_count: dictionary containing asset classes and number of assets
    :return: C matrix used in optimization
    """

    k = len(asset_count.keys())
    n = sum(asset_count.values())

    number_assets = list(asset_count.values())

    C = np.zeros((k,n))

    col_idx = 0
    for i in range(k):
        v = number_assets[i]
        C[i, col_idx: col_idx + v] = 1
        col_idx += v

    return C


def get_Q(df):
    """
    Generate the covariance matrix direclty from the historical returns

    :param df: returns dataframe
    :return: Covariance matrix
    """
    Sigma = df.cov()
    Sigma = Sigma.to_numpy()
    # Sigma = Sigma.T.dot(Sigma)
    return Sigma


def get_exp_rets(df):
    """
    Get expected returns from a geometric average of past returns

    :param df: returns dataframe
    :return: geometric mean returns
    """
    mu = scipy.stats.gmean(1 + df, axis=0) - 1
    return mu




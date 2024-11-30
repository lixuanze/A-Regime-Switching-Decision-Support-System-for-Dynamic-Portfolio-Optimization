# Library imports
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
import tensorflow
import matplotlib.pyplot as plt

# File imports
import optimization_functions as of
import data_processing as dp
from user import User
from portfolio import Portfolio
from factor_models import FactorModel as fm


def run_simulation_dynamic(start_date, end_date, rebalancing_freq, num_periods_rebalancing, person, factor_model,
                           regression_method, min_days_between_rebalance=30, overwrite_allocation=False,
                           overwritten_allocation=None, allow_dynamic_asset_rebalancing=True, reduced_data=False,
                           cardinality=True, transaction_costs=True, regime_window=15, regime_detection=True,
                           overwritten_regime_detection=None, track_return_goal=True, enforce_allocation=True):
    """
    Simulates portfolio performance between two dates by rebalancing periodically within period depending on specified
    rebalancing frequency, and regime detection algorithm.


    :param start_date: start date of backtest: YYYY-MM-DD
    :param end_date: end date of backtest: YYYY-MM-DD
    :param rebalancing_freq: rebalancing frequency for the portfolio: 'M' or 'Q'
    :param num_periods_rebalancing: number of periods of data to include in parameter estimation: int > 0
    :param person: instance of User
    :param factor_model: factor model to be used (PCA, 3 factor, 5 factor, Carhart or LSTM model)
    :param regression_method: Method to be used in factor model regression (LASSO, OLS or Ridge)
    :param min_days_between_rebalance: minimum number of days between rebalancing portfolio
    :param overwrite_allocation: Overwrite allocation given by risk level of user (T/F)
    :param overwritten_allocation: Overwritten allocation vector, b
    :param allow_dynamic_asset_rebalancing: Allow model to change allocation to each asset class depending on risk
    :param reduced_data: Use a data set with fewer assets, increases computational speed
    :param cardinality: Include cardinality constraints in optimization (T/F)
    :param transaction_costs: Include transaction costs in optimization (T/F)
    :param regime_window: number of days of data to be used when calculating the average value of the regime (T/F)
    :param regime_detection: Enable regime detection to give inputs into rebalancing model
    :param overwritten_regime_detection: If regime_detection is False, manually overwrite result returned from regime
    detection for backtesting purposes
    :param track_return_goal: Enable return goal to change risk_level of portfolio (T/F)
    :param enforce_allocation: Enforce allocation by asset class (Cx=b constraint) or let portfolio allocate between
    asset classes in MVO (T/F)
    :return: instance of Portfolio with values populated for backtesting window
    """

    # ---------------- Initiate dates ----------------

    # Convert dates to pd datetime
    t_start = pd.to_datetime(start_date, format='%Y-%m-%d')
    t_end = pd.to_datetime(end_date, format='%Y-%m-%d')

    # Get array of times to rebalance at
    times = pd.date_range(start=t_start, end=t_end, freq=rebalancing_freq, inclusive='both')

    # Get the start date for the training period depending on the rebalancing frequency and number of periods
    if rebalancing_freq == 'Q':
        rebalance_length_days = 91 * num_periods_rebalancing
        t_start_date = t_start - timedelta(days=rebalance_length_days)
    elif rebalancing_freq == 'M':
        rebalance_length_days = 30 * num_periods_rebalancing
        t_start_date = t_start - timedelta(days=rebalance_length_days)

    # Convert start date to string
    training_start = t_start_date.strftime('%Y-%m-%d')

    # Set the start date to the end of the training period
    training_end = start_date

    # Print completed times
    print("Training start : Training end : Test start : Test end")
    print(training_start, training_end, start_date, end_date)

    # ---------------- Load regime detection models ----------------

    # Load model
    classifier_models = load_regime_detection_models()

    # ---------------- Data processing----------------

    # load regime data
    df_regime = pd.read_csv('regime_detection_models/df_regime.csv')
    df_regime['Date'] = pd.to_datetime(df_regime['Date'])
    df_regime.index = df_regime['Date']
    df_regime = df_regime.drop(['Date'], axis=1)

    # regimes
    regimes = ['Free Fall', 'Unstable', 'Weak Up', 'Strong Up']

    # Load data for training and test period
    data = dp.load_data(training_start, end_date, reduced=reduced_data, k=136)  # k=136

    # Get number of assets for each asset class
    asset_count = dp.get_asset_count(data[2])

    # Get list of tickers in model
    tickers = list(data[0].columns.values)
    print(asset_count)

    # Get subset of data used for training
    training_data = dp.get_period_data(data, training_start, training_end)

    # ---------------- Train Regime Model ----------------

    # Get regime features for start of training period
    predicted_regimes = get_regimes(pd.to_datetime(training_start, format='%Y-%m-%d'), classifier_models, df_regime)

    # Set start date for regime moving average as 2x regime_window before start date
    regime_training_date = pd.to_datetime(start_date, format='%Y-%m-%d') - timedelta(days=2 * regime_window)

    # create list of average regimes and regime dates
    trailing_regimes = [predicted_regimes]
    avg_regimes = np.array(trailing_regimes[0])
    regime_dates = [regime_training_date]

    # Run regime prediction on 2x regime_window to get moving average ready. Use this in optimization to get
    # initial weights.
    while regime_training_date < pd.to_datetime(start_date, format='%Y-%m-%d'):

        predicted_regimes = get_regimes(regime_training_date, classifier_models, df_regime)

        # Add predicted regimes to the list
        trailing_regimes.append(predicted_regimes)

        # If length is above threshold, move window across
        if len(trailing_regimes) > regime_window:
            trailing_regimes.pop(0)

            # Take a weighted average with linear weights from 0 to regime_window, weighing latest observation more
            temp_regimes = np.average(np.array(trailing_regimes), axis=0, weights=np.arange(0, regime_window, 1))

            # Add average to list of averages
            avg_regimes = np.vstack((avg_regimes, temp_regimes))
            # Add date
            regime_dates.append(regime_training_date)

        # Increment counter
        regime_training_date += timedelta(days=1)

    # Get actions and new risk level, optimization method from the regime detection
    action, new_risk_level, new_optimization_method = run_regime_detection(0, 'CVaR', avg_regimes[-1, :],
                                                                           regime_detection,
                                                                           overwritten_regime_detection)

    # Compute the allocation per asset class depending on user preferences
    C_true, C_opt, b = compute_allocation(new_risk_level, training_data[2], person,
                                          overwrite_allocation=overwrite_allocation,
                                          overwritten_allocation=overwritten_allocation,
                                          allow_dynamic_asset_rebalancing=allow_dynamic_asset_rebalancing,
                                          enforce_allocation=enforce_allocation)

    # perform initial optimization
    x = optimize_dynamic((new_optimization_method, new_risk_level), training_data, person, factor_model, C_opt, b,
                         transaction_cost=False, first=True, cardinality=cardinality, start_date=training_start,
                         end_date=training_end, regression_method=regression_method)

    print('Initial weights found, portfolio initialized')

    # Convert dates to pandas datetime
    date = pd.to_datetime(start_date, format='%Y-%m-%d') + timedelta(days=1)
    final_date = pd.to_datetime(end_date, format='%Y-%m-%d') - timedelta(days=1)

    # create portfolio
    port = Portfolio(tickers, x, date, np.matmul(C_true, x), initial_value=1000, name='Test Portfolio')

    # Add optimization method and risk level to portfolio
    port.optimization_method.append(new_optimization_method)
    port.risk_level.append(new_risk_level)

    # Create parameters to be used in backtest
    days_remaining = (final_date - date).days
    days_since_rebalance = 0
    last_rebalance_date = start_date
    total_days = days_remaining
    total_return = 0

    # Go through all days in optimization period
    while date < final_date:

        # Get regime features for day at or before current date
        predicted_regimes = get_regimes(date, classifier_models, df_regime)

        # Add predicted regimes to the list
        trailing_regimes.append(predicted_regimes)

        # If length is above threshold, move window across by removing oldest element
        if len(trailing_regimes) > regime_window:
            trailing_regimes.pop(0)

        # Calculate weighted average regime
        temp_regimes = np.average(np.array(trailing_regimes), axis=0, weights=np.arange(0, regime_window, 1))

        # Add moving average and date to list
        avg_regimes = np.vstack((avg_regimes, temp_regimes))
        regime_dates.append(date)
        
        rebalance = False
        
        previous_day = date - timedelta(days=1)
        today_returns = get_period_returns(data, previous_day, date)
        today_returns = today_returns.flatten()
        current_weights = port.weights[-1]
        portfolio_daily_return = np.dot(current_weights, today_returns)
        port.add_daily_return(portfolio_daily_return, date)

        # if enough time has gone since last rebalance, see if rebalance is needed
        if days_since_rebalance >= min_days_between_rebalance:

            # Get current risk level and optimization method from the portfolio
            current_risk_level = port.risk_level[-1]
            current_optimization_method = port.optimization_method[-1]

            # Get actions and new risk level, optimization method from regime_detection
            action, new_risk_level, new_optimization_method = run_regime_detection(current_risk_level,
                                                                                   current_optimization_method,
                                                                                   avg_regimes[-1, :], regime_detection,
                                                                                   overwritten_regime_detection)

            # Rebalance if action or normally scheduled rebalancing
            if action == 'rebalance' or date.strftime("%Y-%m-%d") in times:
                
                rebalance = True
                # Get dates
                period_start_time = date.strftime("%Y-%m-%d")
                period_data_start_time = date - timedelta(days=rebalance_length_days)

                if rebalancing_freq == 'Q':
                    period_end_time = date + pd.tseries.offsets.BQuarterEnd(n=1)
                elif rebalancing_freq == 'M':
                    period_end_time = date + pd.tseries.offsets.BMonthEnd(n=1)

                print("\nTraining start : Period start : Period end")
                print(period_data_start_time, period_start_time, period_end_time)

                print("days since last rebalance: ", days_since_rebalance)

                # ---------------- Return goal calculation ----------------
                # If return goal is allowed to make changes, update optimization method and risk level
                if track_return_goal:
                    new_optimization_method, new_risk_level = return_goal_actions(port, date, final_date, total_days,
                                                                                  avg_regimes[-1, :],
                                                                                  new_risk_level,
                                                                                  new_optimization_method, person)

                # Add to portfolio
                port.risk_level.append(new_risk_level)
                port.optimization_method.append(new_optimization_method)
                print("New optimization method: {}, New risk level: {}".format(new_optimization_method,
                                                                               new_risk_level))
                # ---------------- Perform new Optimization ----------------
                # Get relevant data
                training_data = dp.get_period_data(data, period_data_start_time, period_start_time)
                
                C_true, C_opt, b = compute_allocation(new_risk_level, training_data[2], person,
                                                      overwrite_allocation=overwrite_allocation,
                                                      overwritten_allocation=overwritten_allocation,
                                                      allow_dynamic_asset_rebalancing=allow_dynamic_asset_rebalancing,
                                                      enforce_allocation=enforce_allocation)

                # Optimize for x
                x = optimize_dynamic(port, training_data, person, factor_model, C_opt, b,
                                     transaction_cost=transaction_costs, cardinality=cardinality,
                                     regression_method=regression_method, start_date=period_data_start_time,
                                     end_date=period_start_time)

                port.asset_allocation = np.vstack((port.asset_allocation, np.matmul(C_true, x)))
                port.calculate_turnover(x)
                port.add_new_weights(x)

                # reset days since rebalace
                days_since_rebalance = 0
                last_rebalance_date = period_start_time
        
        
        # Increment days by 1
        date += timedelta(days=1)
        days_since_rebalance += 1


    today_returns = get_period_returns(data, final_date - timedelta(days=1), final_date)
        
    today_returns = today_returns.flatten()
    port.add_new_weights(x)
    current_weights = port.weights[-1]
    
    portfolio_daily_return = np.dot(current_weights, today_returns)
    port.add_daily_return(portfolio_daily_return, final_date)
        
    # Update portfolio last time
    port.risk_level.append(new_risk_level)
    port.optimization_method.append(new_optimization_method)
    port.asset_allocation = np.vstack((port.asset_allocation, np.matmul(C_true, x)))
    port.calculate_turnover(x)
    port.save_to_csv()

    # Add the regimes to the portfolio
    port.regimes = avg_regimes
    port.regime_dates = regime_dates

    return port


def run_regime_detection(current_risk_level, current_optimization_method, avg_regimes, regime_detection,
                         overwritten_regime_detection):
    """
    Get actions from current and predicted regimes, return overwritten regime detection if regime_detection is False

    :param current_risk_level: current portfolio risk level (0,3)
    :param current_optimization_method: current portfolio optimization method (CVaR or MVO)
    :param avg_regimes: Array of latest average regime values
    :param regime_detection: Use regime detection in optimization (T/F)
    :return: list [action, new_risk_level, new_optimization_method]
    """

    if regime_detection:
        # See what the optimal action and risk level/optimization method is depending on the current state and
        # regimes prediction

        actions = get_actions(avg_regimes, current_optimization_method, current_risk_level)
        # actions = get_actions_relative(avg_regimes[-1, 3] - avg_regimes[-1, 0], current_optimization_method,
        #                               current_risk_level)

        return actions
    else:

        return overwritten_regime_detection


def get_actions(average_regimes, current_optimization_method, current_risk_level):
    """
    Given the regimes, current optimization method and current risk level, decide optimal move

    :param current_risk_level: current portfolio risk level (0,3)
    :param current_optimization_method: current portfolio optimization method (CVaR or MVO)
    :param average_regimes: Array of latest average regime values
    :return: list [action, new_risk_level, new_optimization_method]
    """

    action = 'Nothing'
    # If weighted average month prediction over the window is below 0.5, move into lowest risk level
    if average_regimes[3] < 0.75:

        risk_level = 0
        optimization_method = 'CVaR'

        # If current model is not CVaR, rebalance
        if current_optimization_method != 'CVaR':
            action = 'rebalance'

    # If weighted average month prediction over the window is below 1.5, move into 2nd lowest risk level
    elif average_regimes[3] < 1.5:

        risk_level = 1
        optimization_method = 'MVO'

        # Rebalance if current risk is too high
        if current_risk_level >= 1:
            action = 'rebalance'

    # If weighted average month prediction over the window is below 2.5, move into 2nd highest risk level
    elif average_regimes[3] < 2.25:

        risk_level = 2
        optimization_method = 'MVO'

        # Rebalance if too calm or aggressive
        if current_optimization_method == 'CVaR' or current_risk_level != 1:
            action = 'rebalance'

    # If weighted average month prediction over the window is above 2.5, move into highest risk level
    else:
        risk_level = 3
        optimization_method = 'MVO'

        # Rebalance if too calm
        if current_optimization_method == 'CVaR' or current_risk_level < 3:
            action = 'rebalance'

    return action, risk_level, optimization_method


def return_goal_actions(port, date, final_date, total_days, avg_regimes, risk_level, opt_method, person):
    """
    Change the risk level and optimization method based on how portfolio is tracking return goal.

    :param opt_method: current optimization method
    :param port: instance of portfolio
    :param date: current date in backtest (datetime)
    :param final_date: final day of backtest (datetime)
    :param total_days: total duration of backtest (int)
    :param avg_regimes: list of average regime values for current day
    :param risk_level: current portfolio risk level
    :param: person: instance of User
    :return: modified optimization method and risk level as required
    """

    new_optimization_method = opt_method
    new_risk_level = risk_level

    # Get portfolio cumulative return by dividing latest portfolio value by starting value
    portfolio_tot_return = port.value[-1] / port.value[0] - 1

    # Use user's time horizon and annual goal to get total return goal
    total_return_goal = (person.annual_return_goal + 1) ** person.time_horizon - 1

    # Calculate fraction of user's return goal met
    fraction_return_goal_met = portfolio_tot_return / total_return_goal
    print("Fraction of return constraint met: {:.2f}%".format(100 * fraction_return_goal_met))

    # Calculate days elapsed and remaining
    days_remaining = (final_date - date).days
    days_elapsed = total_days - days_remaining

    # Calculate fraction of backtest complete
    fraction_days_complete = 1 - (days_remaining / total_days)

    # Keep track of time elapsed
    print("Fraction complete: {:.2f}%".format(100 * fraction_days_complete))

    # Get annualized portfolio return to date for backtest
    portfolio_geo_ret = (portfolio_tot_return + 1) ** (1 / (days_elapsed / 365)) - 1
            
    # If return goal is met, de-risk.
    if fraction_return_goal_met >= 1:
        print("Return goal met, de-rising portfolio")
        new_risk_level = 0
        new_optimization_method = 'CVaR'

    # If portfolio return/annual return goal < 0.75, increase risk
    elif portfolio_geo_ret / person.annual_return_goal < 0.75:

        # Only change if current method is MVO
        if risk_level >= 1 and avg_regimes[3] > 1.5:
            # Increase risk level by 1
            print("Return not tracking goal, increase risk level to compensate")
            new_risk_level = min(risk_level + 1, 3)

    return new_optimization_method, new_risk_level


def load_regime_detection_models():
    """
    Load regime detection models

    :return: list of regime detection models
    """
    # Load model
    xgb_day = xgb.XGBClassifier()
    xgb_day.load_model('regime_detection_models/next_day_classifier.model')

    xgb_week = xgb.XGBClassifier()
    xgb_week.load_model('regime_detection_models/next_week_classifier.model')

    xgb_month = xgb.XGBClassifier()
    xgb_month.load_model('regime_detection_models/next_month_classifier.model')

    classifier_models = [xgb_day, xgb_week, xgb_month]

    return classifier_models


def get_regimes(date, models, data):
    """
    Get the regimes for the given date using the speficied models and regime data

    :param date: date to use when predicting future regimes
    :param models: list of XGB models used to classify regimes
    :param data: features used by classifier to predict regimes, pandas dataframe
    :return: list of predicted regimes
    """
    # Load models
    xgb_day, xgb_week, xgb_month = models

    # Get regime features for day at or before current date
    iloc_idx = data.index.get_indexer([date], method='pad')
    regime_data = data.iloc[iloc_idx]

    # Current predicted regime
    regime = adjusted_regime(int(regime_data['regime']))

    # Predict regimes in 1 day, 1 week and 1 month
    new_regime_day = adjusted_regime(xgb_day.predict(regime_data)[0])
    new_regime_week = adjusted_regime(xgb_week.predict(regime_data)[0])
    new_regime_month = adjusted_regime(xgb_month.predict(regime_data)[0])

    # Return predicted regimes
    predicted_regimes = [regime, new_regime_day, new_regime_week, new_regime_month]

    return predicted_regimes


def adjusted_regime(regime):
    """
    XGB model order is not increasing. E.g. 0 is the best regime, 2 is the worst etc. Transform this to increasing
    sequence where 0 is the worst and 3 is the best

    :param regime: regime predicted by model
    :return: corrected regime
    """

    if regime == 0:
        return 3
    if regime == 1:
        return 1
    if regime == 2:
        return 0
    if regime == 3:
        return 2


def compute_allocation(risk_level, asset_data, person, overwrite_allocation=False, overwritten_allocation=None,
                       allow_dynamic_asset_rebalancing=True, enforce_allocation=True):
    """
    Given settings and current portfolio risk level, return the desired allocation for portfolio

    :param enforce_allocation: If false, remove constraint to enforce allocation by asset class (T/F)
    :param asset_data: dataframe of assets used and type for each asset
    :param risk_level: current portfolio risk level
    :param overwrite_allocation: Overwrite asset class allocation (T/F)
    :param overwritten_allocation: If overwrite_allocation, the desired new allocation of portfolio
    :param allow_dynamic_asset_rebalancing: Allow asset class allocation to change between optimizations (T/F)
    :return: C_true: actual C matrix, C_opt, C matrix used in optimization, either C_true or 0. b: allocation vector
    """

    # If overwrite allocation, make b vector equal to the allocation given
    if overwrite_allocation:
        b = overwritten_allocation

    else:
        # If dynamic asset rebalancing is allowed, change allocation based on risk level of portfolio
        if allow_dynamic_asset_rebalancing:
            port_params = get_params_from_risk_level(risk_level)
            allocation = port_params['asset_allocation']

        # If dynamic asset rebalancing is not allowed, get allocation from person
        else:
            allocation = person.asset_allocation

        b = list(allocation.values())
        b = np.array(b)

    # Get the number of assets per asset class
    asset_count = dp.get_asset_count(asset_data)

    # Construct allocation matrix from asset counts
    C_true = dp.construct_allocation_matrix(asset_count)

    # Double check assets exist for the asset class
    num_assets = list(asset_count.values())
    for i in range(len(num_assets)):

        # If no assets exist for asset class, set b to 0 for that asset
        if num_assets[i] == 0 and b[i] != 0:
            b[i] = 0

    # Normalize to get b to add up to 1
    b = b / np.sum(b)

    # If allocation is enforced, set the C in optimization equal to C in evaluation
    if enforce_allocation is True:
        # Construct C matrix
        C_opt = C_true

    # If no allocation is enforced, set C to 0 matrix
    else:
        k = len(asset_count.keys())
        n = sum(asset_count.values())

        C_opt = np.zeros((k, n))
        b = np.zeros(k)

    return C_true, C_opt, b


def optimize_dynamic(port, data, person, factor_model, C, b, start_date, end_date, transaction_cost=True, first=False,
                     cardinality=True, regression_method='LASSO'):
    """
    Perform the optimization and return the optimal weights, x

    :param port: instance of portfolio
    :param data: data to be used during optimization (price, share, asset) tuple
    :param person: instance of User
    :param factor_model: factor model to be used
    :param b: allocation to be used
    :param transaction_cost: include transaction cost (T/F)
    :param first: first optimization to find initial weights (T/F)
    :param cardinality: use cardinality (T/F)
    :return: optimal weights, x
    """

    # Get params from portfolio
    if first:
        # If portfolio is not instantiated, load from tuple
        method, risk_level = port
        x_last = None
    else:
        # Get method, risk_level and last weights from portfolio
        method = port.optimization_method[-1]
        risk_level = port.risk_level[-1]
        x_last = port.weights[-1, :]

    # Load data
    price_data, share_data, asset_data = data

    # Load parameters from user and risk level
    params = person.parameters
    port_params = get_params_from_risk_level(risk_level)

    # Get lambda and alpha from risk level
    lambda_ = port_params['lambda']
    alpha = port_params['alpha']

    # Get returns
    returns = dp.get_returns(price_data)

    # Run regression
    if regression_method in ['LASSO', 'Ridge', 'OLS', 'LSTM']:
        # Instantiate factor model
        model = fm(start_date=start_date, end_date=end_date, regression_method=regression_method,
                   factor_returns_method=factor_model, asset_returns=returns)

        # Get mu and sigma from factor model
        mu, Sigma = model.train_factor_model(returns=returns, regression_method=regression_method)
    else:
        # If no factor model is selected, use expected geomean returns
        mu = dp.get_exp_rets(returns)
        Sigma = dp.get_Q(returns)

    # Get parameters
    T = returns.shape[0]
    lb = params['lb']
    ub = params['ub']

    # Optimize
    if method == 'MVO':
        x = of.robust_mvo(mu, Sigma, lambda_, alpha, T, C, b, x_last=x_last, lb=lb, ub=ub, cardinality=cardinality,
                          transaction_penalty=0.01, include_trans_cost=transaction_cost, k_stock=20)

    if method == 'CVaR':
        print("reached CVaR")
        x = of.cvar(alpha, returns, C, b, cardinality=cardinality, ub=ub, lb=lb, k_stock=20,
                    include_trans_cost=transaction_cost, transaction_penalty=0.01, x_last=x_last)

    return x


def get_params_from_risk_level(risk_level):
    """
    Get allocation, lambda and alpha from the portfolio risk level

    :param risk_level: risk level (0,3)
    :return: dict of lambda, alpha and allocation
    """
    if risk_level == 0:
        lambda_ = 100
        alpha = 0.95
        asset_allocation = {
            'alternatives': 0.0,
            'bonds': 0.4,
            'commodities': 0.1,
            'equity_etfs': 0.3,
            'reit': 0.1,
            'individual_stocks': 0.1
        }

    elif risk_level == 1:
        lambda_ = 25
        alpha = 0.95
        asset_allocation = {
            'alternatives': 0.05,
            'bonds': 0.35,
            'commodities': 0.05,
            'equity_etfs': 0.3,
            'reit': 0.05,
            'individual_stocks': 0.2
        }

    elif risk_level == 2:
        lambda_ = 5
        alpha = 0.95
        asset_allocation = {
            'alternatives': 0.05,
            'bonds': 0.25,
            'commodities': 0.05,
            'equity_etfs': 0.2,
            'reit': 0.1,
            'individual_stocks': 0.35
        }

    elif risk_level == 3:
        lambda_ = 1
        alpha = 0.9
        asset_allocation = {
            'alternatives': 0.1,
            'bonds': 0.1,
            'commodities': 0.0,
            'equity_etfs': 0.2,
            'reit': 0.1,
            'individual_stocks': 0.5
        }

    return {'lambda': lambda_, 'alpha': alpha, 'asset_allocation': asset_allocation}


def get_period_returns(data, start_date, end_date):
    """
    Return the returns for a period given the price table and start and end dates

    :param data: (price_data, share_data, asset_data) tuple
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :return: ndarray (n x 1)
    """
    price_data = data[0]

    start_price = price_data.iloc[price_data.index.get_indexer([start_date], method='pad')[0]]

    end_price = price_data.iloc[price_data.index.get_indexer([end_date], method='pad')[0]]

    start_price = start_price.to_numpy()
    end_price = end_price.to_numpy()

    period_returns = np.divide(end_price, start_price) - 1

    return period_returns


def plot_regimes(dates, regimes):
    """
    Plot portfolio regimes

    :param dates: list of dates
    :param regimes: list of regimes for those dates
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, regimes[:, 0], dates, regimes[:, 3])
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Regimes")
    # plt.legend(['current', '1d', '1w', '1m'])
    plt.legend(['current', '1m'])
    plt.show()


def plot_regime_difference(dates, difference):
    """
    Plot difference between 1m prediction and current regime

    :param dates: list of dates
    :param difference: list of difference
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, difference)
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Difference")
    plt.legend(['1m - current'])
    plt.show()


def compare_results(port, start_date, end_date, allocation=np.array([0.3, 0.7]), filename="comparison_results.csv"):
    """
    Compare the portfolio wealth evolution with the 70/30 index and save results.

    :param port: portfolio
    :param start_date: backtest start date
    :param end_date: backtest end date
    :param allocation: allocation of stocks and bonds, by default 70/30
    :param filename: name of the CSV file to save results
    :return:
    """
    # Get dates from portfolio
    dates = port.time

    # Load only BND and SPY for these dates
    data = dp.load_data(start_date, end_date, selected_tickers=['BND', 'SPY'])

    # Get initial portfolio wealth
    index_value = [port.value[0]]

    # Iterate through all rebalancing dates in portfolio and evaluate index wealth
    for i in range(len(dates) - 1):
        # Get dates
        start = dates[i]
        end = dates[i + 1]

        # Get returns for the period
        rets = get_period_returns(data, start, end)

        # Weight the returns by the allocation
        weighted_return = np.matmul(rets, allocation)

        # Get new wealth of portfolio
        new_value = index_value[i] * (1 + weighted_return)

        # Add new wealth to array
        index_value.append(new_value)

    # Print the total index return over backtest
    print("Index return: {:.2f}%".format(100 * (index_value[-1] / index_value[0] - 1)))

    # Save results to CSV
    results_data = {
        "Time": dates,
        "Portfolio Value": port.value,
        "Index Value": index_value
    }
    df = pd.DataFrame(results_data)
    df.to_csv(filename, index=False)
    print(f"Comparison results saved to {filename}.")


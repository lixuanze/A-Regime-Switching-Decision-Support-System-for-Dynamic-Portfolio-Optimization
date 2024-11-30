import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import scipy.stats
matplotlib.use('Agg')


class Portfolio:
    """
    We need a portfolio class to keep track how our portfolio is doing. This class achieves this goal.
    """

    def __init__(self, tickers, initial_weights, starting_time, initial_allocation, initial_value=1, name=None):
        """
        Portfolio Class Initialization.
        """
        self.num_assets = len(tickers)
        self.tickers = tickers
        self.weights = initial_weights.reshape((1, self.num_assets))
        self.value = np.array([initial_value])
        self.returns = np.zeros((1,self.num_assets))
        self.asset_allocation = initial_allocation.reshape((1, 6))
        self.total_returns = np.array([0])
        self.time = [starting_time]
        self.name = name
        self.risk_level = []
        self.optimization_method = []
        self.turnover = [0]
        self.regime_dates = None
        self.regimes = None
        self.daily_returns = []
        self.cumulative_returns = [0]


    def add_new_weights(self, new_weights):
        """
        This method adds new weights when a new period is on.
        """
        self.weights = np.vstack((self.weights, new_weights))


    def add_period_returns(self, new_returns):
        """
        This method add new returns when new return is on.
        """
        self.returns = np.vstack((self.returns, new_returns))

    def add_cumulative_portfolio_return(self, new_return):
        """
        This method adds new total portfolio returns and calculates the portfolio value for the period.
        """
        self.total_returns = np.append(self.total_returns, new_return)

        value = self.value[-1] * (new_return + 1)
        self.value = np.append(self.value, value)
    
    def add_daily_return(self, daily_return, current_date):
        """
        Add daily portfolio return and update portfolio value.
    
        :param daily_return: Portfolio return for the day (float)
        :param current_date: Current date as a datetime object
        """
        self.daily_returns.append(daily_return)
        # Update cumulative return: (1 + r1) * (1 + r2) * ... - 1
        cumulative = (1 + np.array(self.daily_returns)).prod() - 1
        self.cumulative_returns.append(cumulative)
        # Update portfolio value
        new_value = self.value[-1] * (1 + daily_return)
        self.value = np.append(self.value, new_value)
        # Update time
        self.time.append(current_date)

    def calculate_drawdowns(self):
        """
        Calculate drawdowns over time.

        :return: Numpy array of drawdowns.
        """
        running_max = np.maximum.accumulate(self.value)
        drawdowns = (self.value - running_max) / running_max
        return drawdowns

    def get_max_drawdown(self):
        """
        Calculate the Maximum Drawdown (MDD).

        :return: Maximum Drawdown as a float (e.g., -0.25 for 25% drawdown)
        """
        drawdowns = self.calculate_drawdowns()
        max_drawdown = np.min(drawdowns)
        return max_drawdown
        
    def update_time(self, time_):
        """
        This method update the time we need to rebalance the portfolio.
        """
        self.time.append(time_)

    def calculate_turnover(self, new_weights):
        """
        Calculate portfolio turnover and add to portfolio
        :param new_weights: new x vector
        """
        turnover = np.sum(np.abs(self.weights[-1, :] - new_weights))
        self.turnover.append(turnover)
        print(self.turnover)

    def plot_value(self):
        """
        Plot the cumulative returns for the portfolio
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.value)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Portfolio's Cumulative Return")
        plt.savefig("total_return.png", bbox_inches='tight', dpi=300)
        plt.show()

    def plot_weights(self):
        """
        Plot the changing weights for the portfolio
        """
        plt.figure(figsize=(12, 6))
        plt.stackplot(self.time, self.weights.T)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Time")
        plt.ylabel("Individual asset weights")
        plt.show()

    def get_portfolio_snapshot(self, time):
        """
        Get weights of portfolio at certain time

        :param time:
        :return: weights at time t
        """
        try:
            idx = self.time.index(time)

        except ValueError:
            return None

        weights = self.weights[idx]

        return weights

    def plot_method(self):
        """
        plot the optimization method used

        """

        method = [0 if i=='CVaR' else 1 for i in self.optimization_method]

        plt.figure(figsize=(12, 6))
        plt.plot(self.time, method)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Optimization method (1 - MVO, 0 - CVaR")
        plt.show()

    def plot_risk_level(self):
        """
        Plot the portfolio risk level over time

        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.risk_level)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Risk level")
        plt.show()

    def plot_asset_allocation(self):
        """
        plot the portfolio asset class allocation over time

        """
        plt.figure(figsize=(12, 6))
        plt.stackplot(self.time, self.asset_allocation.T)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Time")
        plt.ylabel("Asset class weights")
        plt.legend(['Alternatives', 'Bonds', 'Commodities', 'Equity ETFs', 'REITs', 'Individual stocks'])
        plt.show()

    def get_performance_statistics(self):
        """
        Return the Sharpe Ratio and average turnover for the portfolio throughout the tested period.
    
        :return: Tuple of Sharpe Ratio (SR) and average turnover.
        """
        print(len(self.daily_returns))
        df = pd.DataFrame({
            'Date': self.time,
            'Portfolio Value': self.value
        })

        df_cleaned = df.drop_duplicates(subset='Portfolio Value', keep='first')
        df_cleaned = df_cleaned.sort_values('Date').reset_index(drop=True)

        self.time = df_cleaned['Date'].tolist()
        self.value = df_cleaned['Portfolio Value'].values
        
        daily_returns = np.diff(self.value) / self.value[:-1]
        
        geo_mean = np.prod(1 + daily_returns) ** (252 / len(daily_returns)) - 1  # Annualized
        volatility = np.std(daily_returns) * np.sqrt(252)
        SR = geo_mean / volatility if volatility != 0 else 0
            
        avg_turnover = np.sum(self.turnover)/(48)
        
        max_drawdown = self.get_max_drawdown()
        
        return SR, avg_turnover, max_drawdown

    
    def save_to_csv(self, filename="portfolio_performance.csv"):
        """
        Save the portfolio's performance data to a CSV file.
        """
        if len(self.turnover) < len(self.time):
            padding = [0] * (len(self.time) - len(self.turnover))
            self.turnover = self.turnover + padding
        else:
            self.turnover = self.turnover[:len(self.time)]
        data = {
            "Time": self.time,
            "Portfolio Value": self.value,
            "Turnover": self.turnover,
        }
        print(len(self.time), len(self.value), len(self.turnover))
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Portfolio performance saved to {filename}.")
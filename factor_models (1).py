from math import sqrt
import pandas as pd
import urllib.request
import zipfile
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

class FactorModel:
    """
    This class is a combo for all factor models we explored and implemented. This includes: The CAPM, the Fama-French
    3 Factors, the Carhart 4 Factors, the PCA, and depending on the progress, some Deep Learning Methods/NLP Methods
    will be implemented.

    This class uses multiple regression & deep learning methods to estimate parameters (i.e mu & Q) for the factor
    models used.
    """

    def __init__(self, start_date, end_date, asset_returns, regression_method='LSTM',
                 factor_returns_method='Fama_French_5_factors'):
        """
        Initiating the Factor Model Class

        :param start_date: start date, pandas datetime
        :param end_date: end date, pandas datetime
        :param asset_returns: Return data for assets
        :param regression_method: Regression method to use
        :param factor_returns_method: factor model to use
        """
        self.regression_method = regression_method
        self.start_date = start_date
        self.end_date = end_date
        self.factor_returns = self.get_factor_returns(factor_returns_method, asset_returns)[start_date: end_date]

    def get_factor_returns(self, factor, asset_returns):
        """
        This method return the factor returns from the input dataset.

        :param factor: factor model to use
        :param asset_returns: asset returns or PCA
        ":return factor_returns: returns for the factor model
        """
        if factor == 'CAPM':
            factor_returns = self.get_CAPM_returns()
        elif factor == 'Fama_French_3_factors':
            factor_returns = self.get_Fama_French_returns_3_factors()
        elif factor == 'Carhart':
            factor_returns = self.get_Carhart_returns()
        elif factor == 'Fama_French_5_factors':
            factor_returns = self.get_Fama_French_returns_5_factors()
        elif factor == 'PCA':
            factor_returns = self.get_PCA_returns(asset_returns)
        return factor_returns

    def get_CAPM_returns(self):
        """
        This method simply get CAPM period returns.

        :return CAPM returns
        """
        # Return first row if PCA returns
        return self.get_Fama_French_returns_3_factors().iloc[:, 0]

    def get_Fama_French_returns_3_factors(self):
        """
        This method download the fama-french 3 factor model's returns.

        :return Fama French 3 Factor returns
        """
        # Data Preprocessing
        # Try to load from CSV, if not, download from site
        try:
            returns = pd.read_csv('factor_models/FF_3_factor.csv', index_col=0)

        except FileNotFoundError:

            link = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
            urllib.request.urlretrieve(link, 'fama_french.zip')
            zip = zipfile.ZipFile('fama_french.zip', 'r')
            zip.extractall()
            zip.close()

            Fama_French_returns = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, index_col=0)
            Fama_French_returns = Fama_French_returns[:-1].copy()
            Fama_French_returns.index = pd.to_datetime(Fama_French_returns.index, format="%Y%m%d")
            Fama_French_returns.to_csv("FF_3_factor.csv")

            returns = Fama_French_returns.copy()

        returns.index = pd.to_datetime(returns.index)

        # Set correct scale
        returns['Mkt-RF'] = returns['Mkt-RF'].div(100)
        returns['SMB'] = returns['SMB'].div(100)
        returns['HML'] = returns['HML'].div(100)

        # Remove risk free rate
        returns = returns.drop(['RF'], axis=1)

        return returns

    def get_Carhart_returns(self):
        """
        This method simply get Carhart 4 Factor Model period returns.

        :return Carhart returns
        """
        try:
            momentum_returns = pd.read_csv('factor_models/FF_momentum_factor.csv', index_col=0)

        # Download from web if files don't exist
        except FileNotFoundError:
            link = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

            urllib.request.urlretrieve(link, 'momentum.zip')
            zip = zipfile.ZipFile('momentum.zip', 'r')
            zip.extractall()
            zip.close()

            momentum_factor = pd.read_csv('F-F_Momentum_Factor_daily.CSV', skiprows=13, index_col=0)
            momentum_factor = momentum_factor[:-1].copy()
            momentum_factor.index = pd.to_datetime(momentum_factor.index, format='%Y%m%d')
            momentum_factor.to_csv("FF_momentum_factor.csv")

            momentum_returns = momentum_factor.copy()

        momentum_returns.index = pd.to_datetime(momentum_returns.index)

        # Scale correclty
        momentum_returns.iloc[:, 0] = momentum_returns.iloc[:, 0].div(100)

        # Get the 3 factor returns and add the momentum factor to get the carhart factor returns
        Carhart_returns = pd.concat([self.get_Fama_French_returns_3_factors(), momentum_returns], axis=1).dropna()
        return Carhart_returns

    def get_Fama_French_returns_5_factors(self):
        """
        This method download the fama-french 5 factor model's returns.

        :return factor returns
        """

        try:
            returns = pd.read_csv('factor_models/FF_5_factor.csv', index_col=0)

        # Download from web if file doesn't exist
        except FileNotFoundError:

            link = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

            urllib.request.urlretrieve(link, 'fama_french_5.zip')
            zip = zipfile.ZipFile('fama_french_5.zip', 'r')
            zip.extractall()
            zip.close()
            Fama_French_returns = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.CSV', skiprows=3, index_col=0)

            Fama_French_returns.index = pd.to_datetime(Fama_French_returns.index, format='%Y%m%d')

            Fama_French_returns.to_csv("FF_5_factor.csv")

            returns = Fama_French_returns.copy()

        returns.index = pd.to_datetime(returns.index)

        # Scale correclty
        returns['Mkt-RF'] = returns['Mkt-RF'].div(100)
        returns['SMB'] = returns['SMB'].div(100)
        returns['HML'] = returns['HML'].div(100)
        returns['RMW'] = returns['RMW'].div(100)
        returns['CMA'] = returns['CMA'].div(100)

        returns = returns.drop(['RF'], axis=1)

        return returns

    def get_PCA_returns(self, asset_returns):
        """
        This is a PCA implementation by hand.

        :return factor returns
        """
        # Get returns in numpy form
        excess_returns = asset_returns
        returns = excess_returns.to_numpy()

        T = returns.shape[0]

        # Standardize returns
        xQx = (returns - returns.mean(axis=0)).transpose()

        # Get the Sigma
        sigma = 1 / (T - 1) * np.matmul(xQx, xQx.transpose())

        # Perform decomposition
        eigval, eigvec = np.linalg.eig(sigma)

        # Get the principal components
        principal_components = np.matmul(eigvec.transpose(), xQx).transpose()

        # Extract PCA factors
        pca_factors = np.real(principal_components[:, 0:10])

        # make into dataframe and assign index
        pca_returns = pd.DataFrame(pca_factors)
        pca_returns.index = asset_returns.index

        return pca_returns

    def train_factor_model(self, returns, regression_method, alpha=0.1):
        """
        This method simply takes a method and return an estimated mu and Q based on the user's choice.

        :param returns: asset returns dataframe
        :param regression_method: regression method to use
        :param alpha: Alpha to use in LASSO and Ridge regression
        :return mu and Sigma
        """
        if regression_method == 'OLS' or regression_method == 'LASSO' or regression_method == 'Ridge':
            return self.normal_factor_model(returns, regression_method, alpha)
        elif regression_method == 'CNN':
            return self.CNN(returns)
        else:
            return self.LSTM(returns)
    
    def CNN(self, stock_returns):
        """
        Run CNN model to predict mu

        :param stock_returns: asset returns
        :return: mu, Sigma
        """
        # Get factor returns
        factor_returns = self.factor_returns

        # Get data frame indexes
        factor_index = factor_returns.index.tolist()
        assets_index = stock_returns.index.tolist()

        # Get common dates
        common_times = list(set(factor_index).intersection(assets_index))
        common_times.sort()

        # Get data frames with common dates
        stock_returns = stock_returns.loc[common_times]
        factor_returns = factor_returns.loc[common_times]

        # Split data into test and train windows
        X_train, X_test, y_train, y_test = train_test_split(
            factor_returns.to_numpy(), stock_returns.to_numpy(), test_size=0.2, random_state=123
        )

        # Reshape data for CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define the CNN model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(y_train.shape[1]))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=50, verbose=1)

        # Predict mu using the CNN model
        mu_CNN = model.predict(X_test)

        # Get sigma from normal factor model
        _, sigma = self.normal_factor_model(stock_returns, 'LASSO')

        # Evaluate model performance
        mse = ((y_test - mu_CNN) ** 2).mean()
        print(f"Mean Squared Error (CNN): {mse}")

        return mu_CNN[-1,:], sigma
        
    def normal_factor_model(self, asset_returns, regression_method, alpha=0.1):
        """
        This method performs OLS, LASSO or Ridge regression depending on the preference specified

        :param asset_returns: asset returns
        :param regression_method: regression method to use
        :param alpha: alpha for LASSO and Ridge
        :return: mu, Sigma
        """

        # Get factor returns
        factor_returns = self.factor_returns

        # Get data frame indexes
        factor_index = factor_returns.index.tolist()
        assets_index = asset_returns.index.tolist()

        # Get common dates
        common_times = list(set(factor_index).intersection(assets_index))
        common_times.sort()

        # Get data frames with common dates
        asset_returns = asset_returns.loc[common_times]
        factor_returns = factor_returns.loc[common_times]

        num_periods = len(common_times)

        # Training
        F = factor_returns.cov()
        F = F.to_numpy()

        f_bar = stats.gmean(factor_returns + 1, axis=0) - 1

        # Initiate Linear Regression Variables and responses
        X = factor_returns.to_numpy()
        y = asset_returns.to_numpy()

        # Create proper data matrix
        X = np.hstack((np.ones((num_periods, 1)), X))

        # Model fitting with scikit-learn
        if regression_method == 'OLS':
            model = LinearRegression(fit_intercept=False).fit(X, y)

        elif regression_method == 'LASSO':
            model = Lasso(alpha=alpha).fit(X, y)

        elif regression_method == 'Ridge':
            model = Ridge(alpha=alpha).fit(X, y)

        # Extract coefficients
        B = model.coef_.T

        # Get intercepts and factor loadings
        alpha = B[0, :]
        V = B[1:, :]

        # Calculate expected returns
        mu = alpha + np.matmul(V.T, f_bar)

        # Get dimensions of factor returns
        N = factor_returns.shape[0]
        p = factor_returns.shape[1]

        # Get residuals
        ep = y - np.matmul(X, B)
        norms = np.linalg.norm(ep, axis=0)
        D = 1 / (N - p - 1) * np.diag(norms)

        # Construct Q matrix
        Q = np.matmul(V.T, np.matmul(F, V)) + D

        return mu, Q

    def LSTM(self, stock_returns):
        """
        Run LSTM model to predict mu

        :param stock_returns: asset returns
        :return: mu, Sigma
        """
        # Getfactor returns
        factor_returns = self.factor_returns

        # Get data frame indexes
        factor_index = factor_returns.index.tolist()
        assets_index = stock_returns.index.tolist()

        # Get common dates
        common_times = list(set(factor_index).intersection(assets_index))
        common_times.sort()

        # Get data frames with common dates
        stock_returns = stock_returns.loc[common_times]
        factor_returns = factor_returns.loc[common_times]

        # Split data into test and train windows
        X_train, X_test, y_train, y_test = self.LSTM_preprocessing(factor_returns, stock_returns,
                                                                   0.7 * len(factor_returns),
                                                                   0.2 * len(factor_returns))

        # Min-Max Scaler
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

        # Get the mu estimate using LSTM
        mu_LSTM = self.test(X_train, y_train, factor_returns, X_test.shape[1])

        # Get sigma from normal factor model
        mu, sigma = self.normal_factor_model(stock_returns, 'LASSO')

        # Get the score of the model
        score = self.evaluate_forecasts(y_test, mu_LSTM)
        return mu_LSTM[-1,:], sigma

    def LSTM_preprocessing(self, factor_returns, stock_returns, n_train, n_test):
        """
        Get test train split for LSTM network

        :param factor_returns: factor returns
        :param stock_returns: asset returns
        :param n_train: number training rows
        :param n_test:  number testing rows
        :return: train test split
        """
        # Create arrays
        X, y = [], []
        start = 0
        # Create moving windows for training LSTM
        for i in range(len(factor_returns)):
            end = start + int(n_train)
            end_end = end + int(n_test)
            if end_end <= len(factor_returns):
                X.append(factor_returns.iloc[start:end, :])
                y.append(stock_returns.iloc[end:end_end, :].to_numpy())
            start += 1

        # Split moving window arrays into training and testing dataset
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2,
                                                            random_state=123)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the LSTM model and return model instance

        :param X_train: X data
        :param y_train: y data
        :return:
        """
        # TF LSTM Initialization & Training
        # Set up model architecture
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(RepeatVector(y_train.shape[1]))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(y_train.shape[2])))

        # Compile model
        model.compile(loss='mse', optimizer='adam')

        # fit the model to the data
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=True)
        return model

    def evaluate(self, model, testing_data, n_lookback):
        """
        Evaluate model performance and return mu

        :param model: model
        :param testing_data: testing data
        :param n_lookback: number of periods
        :return: mu
        """
        data = np.array(testing_data)
        X_hat = data[-n_lookback:, :]
        X_hat = X_hat.reshape((1, X_hat.shape[0], X_hat.shape[1]))
        y_hat = model.predict(X_hat, verbose=False)
        y_hat = y_hat[0]
        return y_hat

    def evaluate_forecasts(self, y_test, y_hat):
        """
        Evaluate model quality

        :param y_test: actual values
        :param y_hat: predictions
        :return: Mean Squared Error
        """
        # calculate overall RMSE
        mse = ((y_test - y_hat)**2).mean()
        return mse

    def test(self, X_train, y_train, factor_returns, train_periods):
        """
        Train and test model, return mu

        :param X_train: training features
        :param y_train: training targets
        :param factor_returns: factor returns
        :param train_periods: number of training periods
        :return:
        """
        # Train model
        model = self.train(X_train, y_train)

        # Use model to get Mu
        return self.evaluate(model, factor_returns.tail(train_periods), train_periods)
class User:

    def __init__(self, concentration=1, risk_level=0, time_horizon=0, return_goal=0):
        """

        :param concentration: The concentration of the portfolio in individual securities
        :param risk_level: The aggressiveness of the trade-off between risk and reward that a person is willing to
        do. High aggressiveness means promoting higher riskiness at the expense of volatility
        :param time_horizon
        :param return_goal: The person's annual return goal
        """

        self.concentration = concentration  # 0,1,2,3
        self.risk_level = risk_level  # 0,1,2,3
        self.time_horizon = time_horizon
        self.annual_return_goal = return_goal

        self.parameters = self.get_parameters()

        self.asset_allocation = self.parameters['asset_allocation']

    def get_parameters(self):
        """
        get user parameters after altering some things
        :return: user parameters
        """

        if self.time_horizon <= 2:
            # If user has a short investment horizon, reduce aggressiveness to prevent losses
            self.risk_level = min(0, self.risk_level - 1)

            # Limit concentration at medium
            self.concentration = min(1, self.concentration)

        parameters = generate_user_parameters(self)

        return parameters


def generate_user_parameters(user):
    """
    Idea:
    Select appropriate optimization method based on the parameters of the person

    :param user: user instance
    :return: dict of parameters
    """

    # Default parameters
    asset_allocation = {
        'alternatives': 0.05,
        'bonds': 0.3,
        'commodities': 0.1,
        'equity_etfs': 0.3,
        'reit': 0.05,
        'individual_stocks': 0.2
    }

    lambda_ = 10
    alpha = 0.95
    shorting = 0
    lb = 0
    ub = 1
    k_fraction = 0.75


    # Set parameters
    if user.risk_level == 0:
        lambda_ = 100
        alpha = 0.99
        asset_allocation = {
            'alternatives': 0.0,
            'bonds': 0.4,
            'commodities': 0.15,
            'equity_etfs': 0.3,
            'reit': 0.05,
            'individual_stocks': 0.1
        }

    elif user.risk_level == 1:
        lambda_ = 10
        alpha = 0.95
        asset_allocation = {
            'alternatives': 0.05,
            'bonds': 0.3,
            'commodities': 0.1,
            'equity_etfs': 0.3,
            'reit': 0.05,
            'individual_stocks': 0.2
        }

    elif user.risk_level == 2:
        lambda_ = 1
        alpha = 0.9
        asset_allocation = {
            'alternatives': 0.05,
            'bonds': 0.3,
            'commodities': 0.05,
            'equity_etfs': 0.2,
            'reit': 0.1,
            'individual_stocks': 0.3
        }

    elif user.risk_level == 3:
        lambda_ = 1
        alpha = 0.75
        asset_allocation = {
            'alternatives': 0.1,
            'bonds': 0,
            'commodities': 0.1,
            'equity_etfs': 0.1,
            'reit': 0.2,
            'individual_stocks': 0.5
        }

    # Concentration
    if user.concentration == 0:
        shorting = 0

        lb = 0.0005
        ub = 0.1

        k_fraction = 0.75

    elif user.concentration == 1:
        shorting = 1

        lb = 0.0005
        ub = 0.2

        k_fraction = 0.5

    elif user.concentration == 2:
        shorting = 1

        lb = -0.05
        ub = 0.25

        k_fraction = 0.3

    # Final parameters to be returned
    params = {
        'asset_allocation': asset_allocation,
        'lambda': lambda_,
        'alpha': alpha,
        'shorting': shorting,
        'lb': lb,
        'ub': ub,
        'k_fraction': k_fraction,
        'return_goal': user.annual_return_goal
    }

    return params

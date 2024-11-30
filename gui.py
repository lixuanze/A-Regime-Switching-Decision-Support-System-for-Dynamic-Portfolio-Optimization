import numpy as np
from user import User
import dynamic_rebalancing as dr


def run():
    print("\nWelcome to our portfolio optimization software! \n\nThis menu enables you to change configurations and "
          "backtest the portfolios.")

    run = True
    profile_created = False

    current_settings = {
        "overwrite_allocation": False,
        "cardinality": False,
        "transaction_costs": True,
        "dynamic_rebalancing": True,
        "overwritten_allocation": None,
        "factor_model": 'Carhart',
        "regression_method": 'LASSO',
        "num_periods_data": 3
    }

    no_profile_msg = "To start, please create a user profile by entering 'start'"

    menu_msg = "Welcome to the main menu, to view or change your settings, enter 'config'. To backtest the current " \
               "configuration, enter 'backtest'."

    while run:
        if profile_created:
            print(menu_msg)
        else:
            print(no_profile_msg)

        resp = input("> ")

        if resp == 'help':
            pass

        if resp == 'start':
            print("\nPlease enter your preferences to set up a profile")

            person = create_user()

            print("\nWe have this user profile:")

            print_user_profile(person)
            profile_created = True
            allocation = person.asset_allocation

            print("\nUser created. To change settings, enter 'config'. to run optimization backtest, enter 'backtest'")

        if resp == 'config' and profile_created:
            print("Configure menu for the optimization engine")

            current_settings, allocation = change_config(current_settings, allocation)

        if resp == 'backtest' and profile_created:
            backtest(current_settings, person)

        if resp == 'exit':
            run = False


def create_user():
    risk_level = -1

    print("What risk level are you comfortable with? (0 to 3)")
    while (risk_level < 0 or risk_level > 3):

        risk_level = int(input("> "))

        if (risk_level < 0 or risk_level > 3):
            print("Please enter a valid number between 0 and 3")

    concentration = -1
    print("What concentration are you comfortable with? (0 to 3)")
    while (concentration < 0 or concentration > 3):

        concentration = int(input("> "))

        if (concentration < 0 or concentration > 3):
            print("Please enter a valid number between 0 and 3")

    time_horizon = -1
    print("What is the time-horizon of your investment in years?")
    while (time_horizon < 0):

        time_horizon = float(input("> "))

        if (time_horizon < 0):
            print("Please enter positive number")

    return_goal = -1
    print("What is the annual return goal over your investment horizon? Please enter the return goal in decimal form")
    while (return_goal < 0):

        return_goal = float(input("> "))

        if (return_goal < 0):
            print("Please enter positive number")

    person = User(concentration=concentration, risk_level=risk_level, time_horizon=time_horizon,
                  return_goal=return_goal)

    return person


def print_user_profile(person):
    print("Risk level: {}".format(person.risk_level))
    print("Concentration: {}".format(person.concentration))
    print("Time horizon: {} years".format(person.time_horizon))
    print("Return goal: {:.2f}%".format(person.annual_return_goal * 100))


def change_config(config, allocation):
    print("\nCurrent settings:")

    print_info()

    print_settings(config)

    print_allocation(allocation)

    resp = input("Do you want to change any settings? (Y/n): >")

    if resp == 'Y':

        done = False

        while not done:
            print("Please select what you want to change:")
            print("\n1 - Overwrite allocation\n2 - Impose cardinality\n3 - Impose transaction costs\n4 - Allow dynamic "
                  "rebalancing\n5 - Factor Model\n6 - Regression Method\n7 - Periods of data to use")
            print("You can also write 'exit' to exit")
            resp_2 = input("> ")

            if resp_2 == '1':
                new_val = input("enter new value for 'Overwrite Allocation' (T/F): >")
                if new_val == 'T':
                    print("Please enter new allocation:")

                    allocation = update_allocation()
                    config["overwritten_allocation"] = allocation
                    config["overwrite_allocation"] = True
                else:
                    config["overwrite_allocation"] = False
                    config["overwritten_allocation"] = None

            if resp_2 == '2':
                new_val = input("enter new value for 'Impose cardinality' (T/F): >")

                if new_val == 'T':
                    config['cardinality'] = True
                else:
                    config['cardinality'] = False

            if resp_2 == '3':
                new_val = input("enter new value for 'Impose transaction costs' (T/F): >")

                if new_val == 'T':
                    config['transaction_costs'] = True
                else:
                    config['transaction_costs'] = False

            if resp_2 == '4':
                new_val = input("enter new value for 'Allow dynamic rebalancing' (T/F): >")

                if new_val == 'T':
                    config['dynamic_rebalancing'] = True
                else:
                    config['dynamic_rebalancing'] = False

            if resp_2 == '5':
                new_val = input("enter new value for 'Factor Model' (CAPM', 'Fama_French_3_factors', "
                                "'Fama_French_5_factors', 'Carhart', or 'PCA'.): >")
                config['factor_model'] = new_val

            if resp_2 == '6':
                new_val = input("enter new value for 'Regression Method' ('LASSO', 'Ridge', 'OLS' or 'LSTM'): >")
                config['regression_method'] = new_val

            if resp_2 == '7':
                new_val = input("enter new value for 'Number of periods of data' (int > 0): >")
                config['num_periods_data'] = new_val

            if resp_2 == 'exit':
                done = True

        print("Updated settings are:")
        print_settings(config)


        print_allocation(allocation)

    else:
        print("No changes made")

    return config, allocation


def print_info():
    print("Overwrite allocation: (T/F) Overwrite the allocation selected by regime detection")
    print("Allow dynamic rebalancing: (T/F) Allow asset allocation to switch throughout backtest")


def print_settings(config):
    print("Overwrite allocation:\t\t{}".format(config["overwrite_allocation"]))
    print("Impose cardinality:\t\t\t{}".format(config["cardinality"]))
    print("Impose transaction costs:\t{}".format(config["transaction_costs"]))
    print("Allow dynamic rebalancing:\t{}".format(config["dynamic_rebalancing"]))
    print("Factor Model:\t\t\t\t{}".format(config["factor_model"]))
    print("Regression Method:\t\t\t{}".format(config["regression_method"]))
    print("Periods of data:\t\t\t{}".format(config["num_periods_data"]))


def print_allocation(allocation):
    print("\nCurrent allocation:")
    print("\tAlternatives:\t\t{:.2f}%".format(100 * allocation['alternatives']))
    print("\tBonds:\t\t\t\t{:.2f}%".format(100 * allocation['bonds']))
    print("\tCommodities:\t\t{:.2f}%".format(100 * allocation['commodities']))
    print("\tEquity ETFs:\t\t{:.2f}%".format(100 * allocation['equity_etfs']))
    print("\tREITs:\t\t\t\t{:.2f}%".format(100 * allocation['reit']))
    print("\tIndividual stocks:\t{:.2f}%".format(100 * allocation['individual_stocks']))


def update_allocation():
    alternatives = float(input("Alternatives: >"))
    bonds = float(input("Bonds: >"))
    commodities = float(input("Commodities: >"))
    equity_etfs = float(input("Equity ETFs: >"))
    reits = float(input("REITs: >"))
    individual_stocks = float(input("Individual Stocks: >"))

    allocation = np.array([alternatives, bonds, commodities, equity_etfs, reits, individual_stocks])

    allocation = allocation / np.sum(allocation)

    print(allocation)

    asset_allocation = {
        'alternatives': allocation[0],
        'bonds': allocation[1],
        'commodities': allocation[2],
        'equity_etfs': allocation[3],
        'reit': allocation[4],
        'individual_stocks': allocation[5]
    }

    return asset_allocation


def backtest(config, person):
    start_date = input("Enter start date (YYYY-MM-DD): >")
    end_date = input("Enter end date (YYYY-MM-DD): >")

    rebalancing_freq = 'Q'
    rebalancing_window = config['num_periods_data']
    regression_method = config['regression_method']
    factor_model = config['factor_model']
    regime_window = 30

    overwrite_asset_allocation = config['overwrite_allocation']
    allow_dynamic_asset_rebalancing = config['dynamic_rebalancing']
    allocation = config['overwritten_allocation']
    cardinality = config['cardinality']
    transaction_costs = config['transaction_costs']

    if allocation is None:
        allocation = person.asset_allocation

    b = list(allocation.values())
    b = np.array(b)

    print("\nStarting backtest")

    port = dr.run_simulation_dynamic(start_date, end_date, rebalancing_freq, rebalancing_window, person,
                                     factor_model, regression_method, min_days_between_rebalance=40,
                                     overwrite_allocation=overwrite_asset_allocation,
                                     overwritten_allocation=b,
                                     allow_dynamic_asset_rebalancing=allow_dynamic_asset_rebalancing, reduced_data=True,
                                     cardinality=cardinality, transaction_costs=transaction_costs,
                                     regime_window=regime_window)


    port_SR, port_turnover = port.get_performance_statistics()
    print("SR and Turnover: {:.2f}%, {:.2f}%".format(100 * port_SR, 100 * port_turnover))

    print("Total return: {:.2f}%".format(100 * (port.value[-1] / port.value[0] - 1)))

    dr.plot_regimes(port.regime_dates, port.regimes)
    # plot_regime_difference(port.regime_dates, port.regimes[:, 3] - port.regimes[:, 0])

    dr.compare_results(port, start_date, end_date, allocation=np.array([0.3, 0.7]))

    # port.plot_value()
    # port.plot_weights()
    # port.plot_method()
    port.plot_risk_level()
    port.plot_asset_allocation()


if __name__ == '__main__':
    run()

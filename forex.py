import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# pip install numpy matplotlib scipy yfinance

# We'll get a year's worth of data
# https://finance.yahoo.com/currencies/

data_start = '2023-01-01'
symbol = 'EURUSD=X' # Euro to USD
# symbol = 'JPY=X' # Japanese Yen to USD

data = yf.download(symbol, start=data_start)['Close']
print(data.head())

# We'll turn the data into a numpy array
data = data.to_numpy()
print(data[:5])

# Now we'll define the Ornstein-Uhlenbeck Model
class OUFA:

    def __init__(self, theta=0.1, mu=1.0, sigma=0.1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.z_threshold = 0.01

    def __repr__(self):
        return f"Ornstein-Uhlenbeck process with theta={self.theta}, mu={self.mu}, sigma={self.sigma}"

    def fit(self, data):

        def ou_likelihood(params):

            # We're going to use Euler-Maruyama method to solve the Ornstein-Uhlenbeck process as outlined here: 
            # https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

            # dY(t) = theta * (mu - Y(t)) * dt + sigma * dW(t)
            # Y(0) = data[0]
            # Y(t) = Y(t-1) + theta * (mu - Y(t-1)) * dt + sigma * dW(t)

            theta, mu, sigma = params
            dt = 1
            n = len(data)
            Y = np.zeros(n)

            # * ====================== Write Code Here ====================== * #
            


            # * ============================================================= * #
            
            
            return -0.5 * np.sum((data - Y)**2)

        # We'll use the L-BFGS-B optimization method to minimize the negative log-likelihood
        result = minimize(ou_likelihood, [self.theta, self.mu, self.sigma], method='L-BFGS-B', bounds=((0, None), (None, None), (0, None)))
        self.theta, self.mu, self.sigma = result.x

    def expected_value(self, t, x_0):
        # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#Mathematical_properties

        # * ====================== Write Code Here ====================== * #


        return 0
    
        # * ============================================================= * #

    def calculate_zscore(self, price):
        # ou_mean = self.mu
        ou_mean = self.expected_value(1, price)
        ou_std = max(self.sigma / max(np.sqrt(2 * self.theta), 1e-4), 1e-8)
        z_score = (price - ou_mean) / ou_std
        return z_score
    

# Now we'll do some trading logic

# Big idea:
# We'll take a window of data V(t)
# We'll find the mu and sigma of V(t)
# We'll use that to calculate the z-score of t_0

# If the z-score is above a threshold, we'll short the asset
# If the z-score is below a threshold, we'll long the asset

# We'll use -1 for short, 1 for long, and 0 for close

model = OUFA()
position = 0

'''
Orders should have the format:
{
    'idx: index of the order
    'type': 'long', 'short', 'close'
    'qty': quantity of the order
    'price': price of the order
}
'''
orders = []
window_size = 10
start_idx = window_size + 1

# * ====================== Trade Logic ====================== * #

# This is a simple trading strat
# We'll only buy / sell one unit of the asset
# And only hold one position at a time

order_qty = 1000

for i in range(start_idx, len(data)):

    # Make a window of data
    window = data[i - window_size - 1 : i - 1]

    # * ====================== Write Code Here ====================== * #
    # TODO: Fit the model to the window

    
    # * ============================================================= * #

    # Calculate the z-score
    z_score = model.calculate_zscore(data[i])

    if z_score > model.z_threshold and position != -1:
        orders.append({'idx': i, 'type': 'short', 'qty': order_qty, 'price': float(data[i])})
        position = -1
    elif z_score < -model.z_threshold and position != 1:
        orders.append({'idx': i, 'type': 'long', 'qty': order_qty, 'price': float(data[i])})
        position = 1
    elif abs(z_score) < model.z_threshold:
        orders.append({'idx': i, 'type': 'close', 'qty': order_qty, 'price': float(data[i])})
        position = 0


for order in orders:
    print(order)


# * =============== Portfolio Management =============== * #

capital = 10000
transaction_cost = 0.01     # 1% transaction cost
portfolio = np.zeros(len(data))
portfolio[0] = capital

current_order = orders.pop(0)

for i in range(1, len(data)):
    # If we have an order
    if current_order['idx'] == i:
        if current_order['type'] == 'long':
            capital -= current_order['qty'] * current_order['price'] * transaction_cost
        elif current_order['type'] == 'short':
            capital += current_order['qty'] * current_order['price'] * transaction_cost
        elif current_order['type'] == 'close':
            if current_order['type'] == 'long':
                capital += current_order['qty'] * current_order['price'] * (1 - transaction_cost)
            elif current_order['type'] == 'short':
                capital -= current_order['qty'] * current_order['price'] * (1 - transaction_cost)
        
        if len(orders) > 0:
            current_order = orders.pop(0)

    portfolio[i] = capital


print(f"Initial Capital: {portfolio[0]}")
print(f"Final Capital: {portfolio[-1]}")

# * =============== Plotting =============== * #

# We'll plot the portfolio value and the data in separate subplots

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

ax[0].plot(data, label='Data')
ax[0].set_title('Data')
ax[0].legend()

ax[1].plot(portfolio, label='Portfolio Value', color='red')
ax[1].set_title('Portfolio Value')
ax[1].legend()

plt.show()
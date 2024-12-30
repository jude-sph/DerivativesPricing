import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def trinomial(type_flag, S, X, T, r, b, v, n):
    """
    Constructs a trinomial tree for American option pricing.
    
    Parameters:
    - type_flag (str): "call" for call option, "put" for put option.
    - S (float): Initial stock price.
    - X (float): Strike price of the option.
    - T (float): Time to maturity (in years).
    - r (float): Risk-free interest rate (annualized).
    - b (float): Cost of carry (b = r - q, where q is the dividend yield).
    - v (float): Volatility of the underlying asset (annualized).
    - n (int): Number of time steps in the trinomial tree.
    
    Returns:
    - return_values : Array containing asset prices tree and an Array containing option value, delta, gamma, theta.
    """
    #delta is v here
    #This is a CRR-equivalent trinomial tree, alternative P values in book
    #To discretize GBM jump sizes and probabilities must match the first two moments of the distribution
    #(the mean and variance)
    #assuming up/down/no change
    #up and down jump sizes:
    #u = e ^ (sigma * sqrt (2 * change in t))
    #d = e ^ ( - sigma * sqrt (2 * change in t))
    #probability of going up or down:
    #Pu = big equation as seen on page 300 ... yikes!
    #Pd = 
    #probabilities must sum to unity. Thus, probability of asset price remaining unchanged is:
    #Pm = 1 - Pu - Pd
    #change in t = Time to maturity / n
    #n = number of time steps
    #after asset price tree is built, option value can be found:
    #in the "standard way" using backward induction
    #n >= Int(Tb^2/2sigma^2) + 1 to avoid negative Pm probability
    option_value = np.zeros(2 * n + 1)
    return_values = np.zeros(4)
    #return values goes option value, delta, gamma, theta 
    #asset_prices_tree = []
    if type_flag == "call":
        z = 1
    else:
        z = -1

    dt = T / n
    u = np.exp(v * np.sqrt(2 * dt))
    d = np.exp(-v * np.sqrt(2 * dt))
    pu = np.square((np.exp(b * dt / 2) - np.exp(-v * np.sqrt(dt / 2))) / (np.exp(v * np.sqrt(dt / 2)) - np.exp(-v * np.sqrt(dt / 2))))
    pd = np.square((np.exp(v * np.sqrt(dt / 2)) - np.exp(b * dt / 2)) / (np.exp(v * np.sqrt(dt / 2)) - np.exp(-v * np.sqrt(dt / 2))))
    pm = 1 - pu - pd
    Df = np.exp(-r * dt)

    #asset_prices_tree.append([S])

    for i in range(2 * n):
        option_value[i] = max([0, z * (S * u ** max([0, i - n]) * d ** max([0, n - i]) - X)])
    
    for j in range(n - 1, -1, -1): #from n - 1 to 0 backwards
        for i in range(j * 2):
            option_value[i] = (pu * option_value[i + 2] + pm * option_value[i + 1] + pd * option_value[i]) * Df
            #american specific:
            option_value[i] = max([option_value[i], z * (S * u ** max([0, i - j]) * d ** max([0, j - i]) - X)])
        if j == 1:
            return_values[1] = (option_value[2] - option_value[0]) / (S * u - S * d) #delta?
            return_values[2] = ((option_value[2] - option_value[1]) / (S * u - S) - (option_value[1] - option_value[0]) / (S - S * d)) / (0.5 * (S * u - S * d)) #gamma?
            return_values[3] = option_value[1] #theta?
    print(option_value)
    return_values[3] = (return_values[3] - option_value[0]) / dt / 365 #theta?
    return_values[0] = option_value[0] #option value i think after backwards induction

    return return_values


def trinomial2(type_flag, S, X, T, r, b, v, n):
    """
    Constructs a trinomial tree for American Barrier option pricing and visualizes it.
    
    Parameters:
    - type_flag (str): "call" for call option, "put" for put option.
    - S (float): Initial stock price.
    - X (float): Strike price of the option.
    - T (float): Time to maturity (in years).
    - r (float): Risk-free interest rate (annualized).
    - b (float): Cost of carry (b = r - q, where q is the dividend yield).
    - v (float): Volatility of the underlying asset (annualized).
    - n (int): Number of time steps in the trinomial tree.
    
    Returns:
    - return_values (numpy.ndarray): Array containing option value, delta, gamma, theta.
    """
    # Initialize arrays
    option_value = np.zeros(2 * n + 1)
    return_values = np.zeros(4)
    asset_prices_tree = []  # To store asset prices at each time step
    
    # Determine option type multiplier
    z = 1 if type_flag.lower() == "call" else -1
    
    # Calculate time step
    dt = T / n
    
    # Calculate up and down jump sizes
    u = np.exp(v * np.sqrt(2 * dt))
    d = np.exp(-v * np.sqrt(2 * dt))
    
    # Calculate probabilities
    numerator_pu = np.exp(b * dt / 2) - np.exp(-v * np.sqrt(dt / 2))
    denominator_pu = np.exp(v * np.sqrt(dt / 2)) - np.exp(-v * np.sqrt(dt / 2))
    pu = (numerator_pu / denominator_pu) ** 2
    
    numerator_pd = np.exp(v * np.sqrt(dt / 2)) - np.exp(b * dt / 2)
    pd = (numerator_pd / denominator_pu) ** 2
    
    pm = 1 - pu - pd
    Df = np.exp(-r * dt)
    
    # Initialize asset_prices_tree with initial asset price
    asset_prices_tree.append([S])
    
    # Build asset price tree
    for j in range(1, n + 1):
        level_prices = []
        for i in range(2 * j + 1):
            # Calculate the number of up and down moves
            up_moves = max(i - j, 0)
            down_moves = max(j - i, 0)
            asset_price = S * (u ** up_moves) * (d ** down_moves)
            level_prices.append(asset_price)
        asset_prices_tree.append(level_prices)
    
    # Initialize option values at maturity
    option_value = np.zeros(2 * n + 1)
    for i in range(2 * n + 1):
        asset_price = asset_prices_tree[-1][i]
        option_value[i] = max(0, z * (asset_price - X))
    
    # Backward induction
    for j in range(n - 1, -1, -1):  # From n-1 to 0
        for i in range(2 * j + 1):
            continuation_value = (pu * option_value[i + 2] +
                                  pm * option_value[i + 1] +
                                  pd * option_value[i]) * Df
            asset_price = asset_prices_tree[j][i]
            intrinsic_value = max(0, z * (asset_price - X))
            option_value[i] = max(continuation_value, intrinsic_value)
        # Optionally calculate Greeks here
        if j == 1:
            return_values[1] = (option_value[2] - option_value[0]) / (asset_prices_tree[1][2] - asset_prices_tree[1][0])  # Delta?
            # Gamma calculation corrected for asset price differences
            delta_up = (option_value[2] - option_value[1]) / (asset_prices_tree[1][2] - asset_prices_tree[1][1])
            delta_down = (option_value[1] - option_value[0]) / (asset_prices_tree[1][1] - asset_prices_tree[1][0])
            return_values[2] = (delta_up - delta_down) / ((asset_prices_tree[1][2] - asset_prices_tree[1][0]) / 2)  # Gamma?
            return_values[3] = (option_value[1] - option_value[0]) / dt / 365  # Theta?
    
    # Store option price
    return_values[0] = option_value[0]  # Option value
    
    # Visualize the asset price tree
    visualize_trinomial_tree(asset_prices_tree)
    
    return return_values

def visualize_trinomial_tree(asset_prices_tree):
    """
    Visualizes the trinomial tree using Matplotlib.
    
    Parameters:
    - asset_prices_tree (list of lists): Asset prices at each node for every time step.
    """
    import matplotlib.pyplot as plt
    
    n = len(asset_prices_tree) - 1  # Number of time steps
    plt.figure(figsize=(12, 8))
    
    for j, level in enumerate(asset_prices_tree):
        for i, price in enumerate(level):
            # Calculate position for plotting
            x = j  # Time step on x-axis
            y = price  # Asset price on y-axis
            plt.scatter(x, y, color='blue', s=20)
            if j > 0:
                # Each node can have up to three parents: up, middle, and down
                # Calculate possible parent indices in the previous level
                
                # 'Up' parent (i-th node in previous level)
                if i < len(asset_prices_tree[j - 1]):
                    parent_price_up = asset_prices_tree[j - 1][i]
                    plt.plot([j - 1, j], [parent_price_up, y], color='gray', linewidth=0.5)
                
                # 'Middle' parent ((i-1)-th node in previous level)
                if (i - 1) >= 0 and (i - 1) < len(asset_prices_tree[j - 1]):
                    parent_price_mid = asset_prices_tree[j - 1][i - 1]
                    plt.plot([j - 1, j], [parent_price_mid, y], color='gray', linewidth=0.5)
                
                # 'Down' parent ((i-2)-th node in previous level)
                if (i - 2) >= 0 and (i - 2) < len(asset_prices_tree[j - 1]):
                    parent_price_down = asset_prices_tree[j - 1][i - 2]
                    plt.plot([j - 1, j], [parent_price_down, y], color='gray', linewidth=0.5)
    
    plt.title('Trinomial Tree Asset Price Visualization')
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.show()

#american put option with:
#stock price 100
#strike price 110
#time to maturity 6 months
#RRR = cost-of-carry of 10%
#volatility 27%
#time steps 30
#result = trinomial("put", 100, 110, 0.5, 0.1, 0.1, 0.27, 30)
#result_info = result[1]
#result_tree = result[0]
#visualize_trinomial_tree(result_tree)
result_info = trinomial2("put", 100, 110, 0.5, 0.1, 0.1, 0.27, 30)



print(result_info)
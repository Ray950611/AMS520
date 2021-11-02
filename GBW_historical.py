#%%
import matplotlib.pyplot as plt
import time
import warnings
from datetime import datetime
from functions import *
from scipy.optimize import minimize
warnings.filterwarnings('ignore')


df = pd.read_csv('D:\\SBU Data\\AMS 520\\stock_processed.csv', index_col=0)
df.index = list(map(lambda x: datetime.strptime(x, '%m/%d/%Y').date(), df.index))
df = df[[True]+list(map(lambda x, y: x.month != y.month, df.index[:-1], df.index[1:]))]
sp500 = pd.read_csv('D:\\SBU Data\\AMS 520\\s&p500.csv', index_col=0, thousands=',')
sp500.index = list(map(lambda x: datetime.strptime(x, '%d-%b-%y').date(), sp500.index))
sp500 = sp500[[True]+list(map(lambda x, y: x.month != y.month, sp500.index[:-1], sp500.index[1:]))]

risky_asset_returns = rt_calculate(df).iloc[1:].astype(float)
T, num_risky_assets = risky_asset_returns.shape
t_out = 36

expected_risky_returns = np.tile(risky_asset_returns.iloc[-12-t_out:-t_out].mean().values, (t_out, 1))

init_cash = 1000
dt = 1/12
riskfree_rate = 0.02*dt

num_assets = num_risky_assets + 1
num_steps = t_out
x_vals_init = init_cash * np.ones(num_assets) / num_assets

target_return = 0.3
benchmark_portf = [init_cash * np.exp(dt * target_return)]
for i in range(1, num_steps):
    benchmark_portf.append(benchmark_portf[i-1]*np.exp(dt * target_return))
print(benchmark_portf[0], benchmark_portf[-1])

lambd = 0.001
omega = 1.0
beta = 1000.0
gamma = 0.95
eta = 1.5  # 1.3 # 1.5 # 1.2
rho = 0.4

reward_params = [lambd, omega, eta, rho]

returns = np.c_[riskfree_rate*np.ones((num_steps,1)), risky_asset_returns.values[-num_steps:]]
returns_all = []
returns_all_equal = []
traj = []
x_t = x_vals_init[:]
# Sigma_r = risky_asset_returns.iloc[-24-t_out:-t_out].cov().values
for t in range(num_steps):
    expected_risky_returns[t:] = risky_asset_returns.iloc[-24-t_out+t:-t_out+t].mean().values
    Sigma_r = risky_asset_returns.iloc[-24-t_out+t:-t_out+t].cov().values

    # Create a G-learner
    G_learner = G_learning_portfolio_opt(num_steps,
                                         reward_params,
                                         beta,
                                         benchmark_portf,
                                         gamma,
                                         num_risky_assets,
                                         riskfree_rate,
                                         expected_risky_returns,  # array of shape num_steps x num_stocks
                                         Sigma_r,     # covariance matrix of returns of risky matrix
                                         x_vals_init,  # array of initial values of len (num_stocks+1)
                                         use_for_WM=True)  # use for wealth management tasks

    G_learner.reset_prior_policy()
    error_tol = 1.e-8
    max_iter_RL = 200
    G_learner.G_learning(error_tol, max_iter_RL)

    mu_t = G_learner.u_bar_prior[t,:] + G_learner.v_bar_prior[t,:].mv(torch.tensor(x_t))
    u_t = np.random.multivariate_normal(mu_t.detach().numpy(), G_learner.Sigma_prior[t,:].detach().numpy())
    # compute new values of x_t

    x_next = x_t + u_t

    x_next = (1+returns[t])*x_next
    port_returns = (x_next.sum() - x_t.sum() - np.sum(u_t) - 0.015*np.abs(u_t).sum())/x_t.sum()
    traj.append((x_t, u_t))

    # rename
    x_t = x_next
    returns_all.append(port_returns)
    returns_all_equal.append(returns[t].mean())
    # end the loop over time steps

plt.figure()
plt.plot(returns)

# # Calculate performance of G-learner (Diagnostics only)
SR_G = (np.mean(returns_all)-riskfree_rate*dt)/np.std(returns_all)
SR_G_equal = (np.mean(returns_all_equal)-riskfree_rate*dt)/np.std(returns_all_equal)

r_G = np.array([0]*num_steps, dtype='float64')
r_G_equal = np.array([0]*num_steps, dtype='float64')
for i in range(num_steps):
    r_G[i] += returns_all[i]
    r_G_equal[i] += returns_all_equal[i]

plt.figure()
plt.plot(risky_asset_returns.index[-num_steps:], r_G, color='tab:orange', linewidth=0.9)
plt.plot(risky_asset_returns.index[-num_steps:], r_G_equal, color='tab:blue', linewidth=0.9)
plt.legend(
    ['G-learning: (' + str(np.round(SR_G, 3)) + ')',
     'Equal weight: (' + str(np.round(SR_G_equal, 3)) + ')'], fontsize=8
)
plt.grid(axis='both', linestyle='--', linewidth=0.8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('time', fontsize=9)
plt.ylabel('Portfolio Returns', fontsize=9)



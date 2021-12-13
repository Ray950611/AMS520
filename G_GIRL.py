#%%
import copy
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

import numpy as np

from G_func import *
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')

error_tol = 1.e-8
max_iter_RL = 200
gamma = 0.95
beta = 1000
dt = 1/12
riskfree_rate = 0.02*dt
target_return = 0.8

# # GRIL
def get_loss(trajs,
             num_steps,
             benchmark_portf,
             gamma,
             num_risky_assets,
             riskfree_rate,
             expected_risky_returns,
             Sigma_r,
             x_vals_init,
             max_iter_RL,
             reward_params,
             beta,
             num_trajs,
             grad=False,
             eps=1e-7):


    data_xvals = torch.zeros(num_trajs,  num_steps, num_assets, dtype=torch.float64, requires_grad=False)
    data_uvals = torch.zeros(num_trajs,  num_steps, num_assets, dtype=torch.float64, requires_grad=False)

    for n in range(num_trajs):
        for t in range(num_steps):
            data_xvals[n,t,:] = torch.tensor(trajs[n][t][0],dtype=torch.float64)
            data_uvals[n,t,:] = torch.tensor(trajs[n][t][1],dtype=torch.float64)


    # allocate memory for tensors that wil be used to compute the forward pass
    realized_rewards = torch.zeros(num_trajs, num_steps, dtype=torch.float64, requires_grad=False)
    realized_cum_rewards = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)

    realized_G_fun = torch.zeros(num_trajs, num_steps, dtype=torch.float64, requires_grad=False)
    realized_F_fun  = torch.zeros(num_trajs,  num_steps, dtype=torch.float64, requires_grad=False)

    realized_G_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)
    realized_F_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)

    reward_params_dict={}
    loss_dict={}
    loss_dict[-1]=np.array([0]*len(reward_params), dtype='float64') # perturb up
    loss_dict[1]=np.array([0]*len(reward_params), dtype='float64') # perturb down
    loss_grad = np.array([0]*len(reward_params), dtype='float64')

    if grad: # compute gradient
        for j in range(len(reward_params)):
            for k in [-1,1]:
                reward_params_dict[k]=reward_params
                reward_params_dict[k][j]= reward_params_dict[k][j] + k*eps

                # 1. create a G-learner
                G_learner = G_learning_portfolio_opt(num_steps,
                                                     reward_params_dict[k],
                                                     beta,
                                                     benchmark_portf,
                                                     gamma,
                                                     num_risky_assets,
                                                     riskfree_rate,
                                                     expected_risky_returns,
                                                     Sigma_r,
                                                     x_vals_init,
                                                     use_for_WM=True)

                G_learner.reset_prior_policy()

                # run the G-learning recursion to get parameters of G- and F-functions
                G_learner.G_learning(error_tol, max_iter_RL)

                # compute the rewards and realized values of G- and F-functions from
                # all trajectories
                for n in range(num_trajs):
                    for t in range(num_steps):

                        realized_rewards[n,t] = G_learner.compute_reward_on_traj(t,
                                                                                 data_xvals[n,t,:], data_uvals[n,t,:])

                        realized_G_fun[n,t] = G_learner.compute_G_fun_on_traj(t,
                                                                              data_xvals[n,t,:], data_uvals[n,t,:])

                        realized_F_fun[n,t] = G_learner.compute_F_fun_on_traj(t,
                                                                              data_xvals[n,t,:])

                    realized_cum_rewards[n] = realized_rewards[n,:].sum()
                    realized_G_fun_cum[n] = realized_G_fun[n,:].sum()
                    realized_F_fun_cum[n] = realized_F_fun[n,:].sum()

                loss_dict[k][j] = - beta *(realized_G_fun_cum.sum() - realized_F_fun_cum.sum())
            loss_grad[j]=(loss_dict[1][j]-loss_dict[-1][j])/(2.0*eps)

    G_learner = G_learning_portfolio_opt(num_steps,
                                         reward_params,
                                         beta,
                                         benchmark_portf,
                                         gamma,
                                         num_risky_assets,
                                         riskfree_rate,
                                         expected_risky_returns,
                                         Sigma_r,
                                         x_vals_init,
                                         use_for_WM=True)

    G_learner.reset_prior_policy()

    G_learner.G_learning(error_tol, max_iter_RL)

    # compute the rewards and realized values of G- and F-functions from
    # all trajectories
    for n in range(num_trajs):
        for t in range(num_steps):

            realized_rewards[n,t] = G_learner.compute_reward_on_traj(t,
                                                                     data_xvals[n,t,:], data_uvals[n,t,:])

            realized_G_fun[n,t] = G_learner.compute_G_fun_on_traj(t,
                                                                  data_xvals[n,t,:], data_uvals[n,t,:])

            realized_F_fun[n,t] = G_learner.compute_F_fun_on_traj(t,
                                                                  data_xvals[n,t,:])

        realized_cum_rewards[n] = realized_rewards[n,:].sum()
        realized_G_fun_cum[n] = realized_G_fun[n,:].sum()
        realized_F_fun_cum[n] = realized_F_fun[n,:].sum()

    loss = - beta *(realized_G_fun_cum.sum() - realized_F_fun_cum.sum())
    if grad:
        return loss, loss_grad
    else:
        return loss


def fun(x, trajs, grad_=False, rescale=1.0, constraint=False):
    y=x.copy()
    y[0]/=sc[0]
    y[1]/=sc[1]
    y[2]/=sc[2]
    y[3]/=sc[3]
    with torch.no_grad():
        if grad_==False:
            if constraint:
                y[0] = y[0]**2
                y[1] = 1+y[1]**2
                y[2] = 1+y[2]**2
                y[3] = 1.0/(1.0+np.exp(-y[3]))

            ret = rescale*get_loss(trajs,
                                   num_steps,
                                   benchmark_portf,
                                   gamma,
                                   num_risky_assets,
                                   riskfree_rate,
                                   expected_risky_returns,
                                   Sigma_r,
                                   x_vals_init,
                                   max_iter_RL,
                                   y,
                                   beta,
                                   len(trajs),
                                   grad=grad_,
                                   eps=1e-7).detach().numpy()

            print(ret)
            return ret
        else:
            f, df = get_loss(trajs,
                             num_steps,
                             benchmark_portf,
                             gamma,
                             num_risky_assets,
                             riskfree_rate,
                             expected_risky_returns,
                             Sigma_r,
                             x_vals_init,
                             max_iter_RL,
                             y,
                             beta,
                             len(trajs),
                             grad=grad_,
                             eps=1e-7)
            return f.detach().numpy()*rescale, df*rescale/sc


lambd_0 = 0.001
omega_0 = 1.00
eta_0 = 1.5
beta_0 = beta
rho_0 = 0.40

sc=np.array([1,1,1,1]) # optional re-scaling
x0=np.array([lambd_0, omega_0, eta_0, rho_0])
x0=[np.sqrt(x0[0]), np.sqrt(x0[1]-1), np.sqrt(x0[2]-1), np.log(x0[3]/(1-x0[3]))]
x0*=sc

df = pd.read_csv('D:\\SBU (Local)\\Courses Data\\AMS 520\\stock_processed2.csv', index_col=0)
df.index = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d').date(), df.index))
df = df[[True]+list(map(lambda x, y: x.month != y.month, df.index[:-1], df.index[1:]))]
sp500 = pd.read_csv('D:\\SBU (Local)\\Courses Data\\AMS 520\\s&p500.csv', index_col=0, thousands=',')
sp500.index = list(map(lambda x: datetime.strptime(x, '%d-%b-%y').date(), sp500.index))
sp500 = sp500[[True]+list(map(lambda x, y: x.month != y.month, sp500.index[:-1], sp500.index[1:]))]

risky_asset_returns = rt_calculate(df).iloc[1:].astype(float)
risky_asset_returns = risky_asset_returns.iloc[-50:]
T, num_risky_assets = risky_asset_returns.shape
num_assets = num_risky_assets + 1
window_size = 12

init_cash = 1000
x_vals_init = init_cash*np.ones(num_assets) / num_assets

num_steps = 12
returns_all_equal = []

# reward_params = [lambd, omega, eta, rho]
reward_params = [[0.001, 1.0, 1.5, 0.4]]
reward_params_index = ['Actual']
reward_params2 = [[0.001, 1.0, 1.5, 0.4]]
reward_params_index2 = ['Actual']
reward_params3 = [[0.001, 1.0, 1.5, 0.4]]
reward_params_index3 = ['Actual']
N = [10, 100]

cov_name = 'original'
# cov_name = 'shrinkage'
# cov_name = 'forward'

T = T - window_size
m = T - window_size
equal_weight_portfolio = [init_cash]
g_learner = g_learn(num_steps, num_risky_assets, window_size, x_vals_init, reward_params[0], N[-1])
exp_returns = np.empty((num_steps, num_risky_assets))
for i in range(window_size, T):
    print(risky_asset_returns.index[i])
    returns = risky_asset_returns.iloc[i].values
    exp_returns[(i-window_size) % num_steps:] = risky_asset_returns.iloc[i-window_size:i].mean().values
    if cov_name == 'original':
        sigma_r = risky_asset_returns.iloc[i-window_size:i].cov().values
    elif cov_name == 'shrinkage':
        sigma_r = LedoitWolf().fit(risky_asset_returns.iloc[i-window_size:i].values).covariance_
    elif cov_name == 'forward':
        sigma_r = risky_asset_returns.iloc[i:i+window_size].cov().values

    g_learner = g_learn_rolling(
        (i-window_size) % num_steps, g_learner, exp_returns, sigma_r, returns
    )

    returns_all_equal.append(returns.mean())
    equal_weight_portfolio.append((equal_weight_portfolio[-1]+g_learner.trajs_all[-1][1].sum())*(1+returns_all_equal[-1]))

    if (i-window_size) % num_steps == 0 and i != window_size:
        expected_risky_returns = exp_returns
        Sigma_r = risky_asset_returns.iloc[i-window_size:i].cov().values

        benchmark_portf = [g_learner.x_vals_init.sum() * np.exp(dt * target_return)]
        for j in range(1, num_steps):
            benchmark_portf.append(benchmark_portf[j-1]*np.exp(dt * target_return))

        bnds = ((0.0001*sc[0], 0.0025*sc[0]), (0.01*sc[1], 1.5*sc[1]), (1.001*sc[2],2*sc[2]), (0.01*sc[3],1*sc[3]))
        # Optimize with the Nelder-Mead method
        res = minimize(
            fun, x0, method='Nelder-Mead', args=([trajs[0]], False, 1e-9, True),
            options={'disp': True, 'maxiter':100}, tol=0.01
        )
        reward_params.append([res.x[0]**2/sc[0], (1+res.x[1]**2)/sc[1], (1+res.x[2]**2)/sc[2], 1.0/(1+np.exp(-res.x[3]))/sc[3]])
        reward_params_index.append(risky_asset_returns.index[i-num_steps-1])

        res = minimize(
            fun, x0, method='Nelder-Mead', args=(trajs[0:N[0]], False, 1e-9, True),
            options={'disp': True, 'maxiter':100}, tol=0.01
        )
        reward_params2.append([res.x[0]**2/sc[0], (1+res.x[1]**2)/sc[1], (1+res.x[2]**2)/sc[2], 1.0/(1+np.exp(-res.x[3]))/sc[3]])
        reward_params_index2.append(risky_asset_returns.index[i-num_steps-1])

        res = minimize(
            fun, x0, method='Nelder-Mead', args=(trajs[0:N[1]], False, 1e-9, True),
            options={'disp': True, 'maxiter':100}, tol=0.01
        )
        reward_params3.append([res.x[0]**2/sc[0], (1+res.x[1]**2)/sc[1], (1+res.x[2]**2)/sc[2], 1.0/(1+np.exp(-res.x[3]))/sc[3]])
        reward_params_index3.append(risky_asset_returns.index[i-num_steps-1])

    elif (i-window_size) % num_steps == num_steps-1:
        trajs = copy.deepcopy(g_learner.trajs_multi)

reward_params = pd.DataFrame(
    np.array(reward_params), index=reward_params_index, columns=['lambd', 'omega', 'eta', 'rho']
)
reward_params2 = pd.DataFrame(
    np.array(reward_params2), index=reward_params_index, columns=['lambd', 'omega', 'eta', 'rho']
)
reward_params3 = pd.DataFrame(
    np.array(reward_params3), index=reward_params_index, columns=['lambd', 'omega', 'eta', 'rho']
)

equal_weight_portfolio = equal_weight_portfolio[:-1]

# # Calculate performance of G-learner (Diagnostics only)
SR_G = (np.mean(g_learner.returns_all)-g_learner.riskfree_rate)/np.std(g_learner.returns_all) * np.sqrt(12)
SR_G_equal = (np.mean(returns_all_equal)-g_learner.riskfree_rate)/np.std(returns_all_equal) * np.sqrt(12)

r_G = np.array([0]*m, dtype='float64')
r_G_equal = np.array([0]*m, dtype='float64')
for i in range(m):
    r_G[i] += g_learner.returns_all[i]*12
    r_G_equal[i] += returns_all_equal[i]*12

plt.figure(figsize=(9, 4.5))
plt.plot(reward_params_index[1:],
         [reward_params['lambd'].values[0]]*(reward_params.shape[0]-1), color='tab:orange', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params['lambd'].values[1:], color='tab:blue', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params2['lambd'].values[1:], color='tab:olive', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params3['lambd'].values[1:], color='tab:red', linewidth=0.9)
plt.legend(
    ['lambd: '+str(reward_params['lambd'].values[0]), '1 traj', str(N[0])+' trajs', str(N[1])+' trajs'],
    fontsize=8
)
plot_elegant()
plt.savefig('GIRL lambd ('+cov_name+').png', dpi=500)

plt.figure(figsize=(9, 4.5))
plt.plot(reward_params_index[1:],
         [reward_params['omega'].values[0]]*(reward_params.shape[0]-1), color='tab:orange', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params['omega'].values[1:], color='tab:blue', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params2['omega'].values[1:], color='tab:olive', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params3['omega'].values[1:], color='tab:red', linewidth=0.9)
plt.legend(
    ['omega: '+str(reward_params['omega'].values[0]), '1 traj', str(N[0])+' trajs', str(N[1])+' trajs'],
    fontsize=8
)
plot_elegant()
plt.savefig('GIRL omega ('+cov_name+').png', dpi=500)

plt.figure(figsize=(9, 4.5))
plt.plot(reward_params_index[1:],
         [reward_params['eta'].values[0]]*(reward_params.shape[0]-1), color='tab:orange', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params['eta'].values[1:], color='tab:blue', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params2['eta'].values[1:], color='tab:olive', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params3['eta'].values[1:], color='tab:red', linewidth=0.9)
plt.legend(
    ['eta: '+str(reward_params['eta'].values[0]), '1 traj', str(N[0])+' trajs', str(N[1])+' trajs'],
    fontsize=8
)
plot_elegant()
plt.savefig('GIRL eta ('+cov_name+').png', dpi=500)

plt.figure(figsize=(9, 4.5))
plt.plot(reward_params_index[1:],
         [reward_params['rho'].values[0]]*(reward_params.shape[0]-1), color='tab:orange', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params['rho'].values[1:], color='tab:blue', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params2['rho'].values[1:], color='tab:olive', linewidth=0.9)
plt.plot(reward_params_index[1:], reward_params3['rho'].values[1:], color='tab:red', linewidth=0.9)
plt.legend(
    ['rho: '+str(reward_params['rho'].values[0]), '1 traj', str(N[0])+' trajs', str(N[1])+' trajs'],
    fontsize=8
)
plot_elegant()
plt.savefig('GIRL rho ('+cov_name+').png', dpi=500)

date_index = risky_asset_returns.index[window_size:T]

plt.figure()
plt.plot(date_index, r_G, color='tab:orange', linewidth=0.9)
plt.plot(date_index, r_G_equal, color='tab:blue', linewidth=0.9)
plt.legend(
    ['G-learning: (' + str(np.round(SR_G, 3)) + ')',
     'Equal weight: (' + str(np.round(SR_G_equal, 3)) + ')'], fontsize=8
)
plot_elegant()
plt.title('Portfolio Returns', fontsize=10)
plt.savefig('portfolio returns (lambda='+str(reward_params['lambd'][0])+', '+cov_name+').png', dpi=500)

realized_cum_rewards = g_learner.realized_rewards.sum()
realized_G_fun_cum = g_learner.realized_G_fun.sum()
realized_F_fun_cum = g_learner.realized_F_fun.sum()

realized_rewards = g_learner.realized_rewards.detach().numpy()[1:]
plt.figure()
plt.plot(date_index, realized_rewards, label='realized rewards', color='tab:blue', linewidth=0.9)
plot_elegant()
plt.title('Realized rewards', fontsize=10)
plt.savefig('Realized rewards (lambda='+str(reward_params['lambd'][0])+', '+cov_name+').png', dpi=500)

trajs = g_learner.trajs_all
x_record = []
u_record = []
for i in range(len(trajs)):
    x_record.append(trajs[i][0].sum())
    u_record.append(trajs[i][1].sum())

plt.figure()
plt.plot(date_index, x_record, label='realized portfolio value', color='tab:orange', linewidth=0.9)
plt.plot(date_index, u_record, label='realized cash injection', color='tab:blue', linewidth=0.9)
plt.plot(date_index, equal_weight_portfolio, color='tab:olive', linewidth=0.9)
plt.legend(['portfolio value', 'cash injection', 'equal-weight portfolio value'], fontsize=8)
plot_elegant()
plt.savefig('Realized portfolio value (lambda='+str(reward_params['lambd'][0])+', '+cov_name+').png', dpi=500)

# g_learner.learner.project_cash_injections()
#
# eta_ = g_learner.learner.eta.detach().numpy()
# realized_target_portf = eta_ * g_learner.learner.expected_portf_val.numpy()
#
# plt.figure()
# plt.plot(g_learner.learner.expected_c_t, label='optimal cash installments')
# plt.plot(g_learner.learner.expected_portf_val, label='expected portfolio value')
# plt.plot(realized_target_portf, label='realized target portfolio', color='r')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.title('Optimal cash installment and portfolio value')
# plt.savefig('Cash Installments.png', dpi=500)
#
# print('Optimal Cash Installments:')
# print(g_learner.learner.expected_c_t)




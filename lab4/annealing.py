import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def anneal(x_0, cost_fun, update_fun, decision_fun=None, T_0=2000,
     T_change_fun=None, iters=10**3, max_retries=100, verbose=1, early_stop_fun=None):
    """Simulated annealing. 
    
    Parameters
    ----------
    x_0
        The thing to optimize.
    cost_fun : x -> float
        Cost function.
    update_fun : (x, T) -> x
        Function getting neighbouring state of x.
    decision_fun : T -> bool
        Whether to accept a step towards higher cost.
    T_0 : float (default 2000)
        Initial temperature
    T_change_fun : T -> T (default T = 0.95*T)
        Temperature change at the end of iteration.
    iters: int (default 10**4)
        Iterations.
    max_retries : int (default 100)
        Max tries per iteration to step towards lower cost or accept 
        a step towards higher cost.
    verbose : bool (default true)
        Whether to print updates during the process.
    early_stop_fun : T, cost - > bool
        Function with decision whether to stop the process.

    Returns
    -------
    x
        The optimized thing.
    cost_hist
        Cost history.
    T_hist
        Temperature history.            
   

    """
    x = x_0
    T = T_0
    T_change_fun = T_change_fun or (lambda T: 0.99*T)
    decision_fun = decision_fun or sigmoid_decision
    
    cost_hist = []
    T_hist = []
    last_cost = cost_fun(x)

    if verbose:
        print("Starting...")
        print(f'0%, iter = 0, cost = {round(last_cost,2)}, T = {round(T,2)}')
    
    for i in range(10):
        for _ in range(iters//10):
            for _ in range(max_retries):
                new_x = update_fun(x, T)
                cost = cost_fun(new_x)
                if cost<last_cost or decision_fun(T):
                    x = new_x
                    last_cost = cost
                    break
            cost_hist.append(last_cost)
            T_hist.append(T)
            T = T_change_fun(T)
        if verbose:
            print(f'{(i+1)*10}%, iter = {(i+1)*iters//10}, cost = {round(last_cost,2)}, T = {round(T,2)}')
        if early_stop_fun and early_stop_fun(T, last_cost):
            print("Stopping!")
            return x, cost_hist, T_hist
    if verbose:
        print("Finished!")

        
    return x, cost_hist, T_hist

def sigmoid_decision(T):
    return np.tanh((T-750)/500)>2*np.random.random()-0.9

def plot_annealing(cost_hist, T_hist):
    assert len(cost_hist) == len(T_hist)

    X = list(range(len(cost_hist)))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=X, y=cost_hist, name="Cost"), secondary_y=False)
    fig.add_trace(go.Scatter(x=X, y=T_hist, name="Temperature"), secondary_y=True)
    fig.update_xaxes(title_text="Iterations")
    fig.update_yaxes(title_text="Cost", secondary_y=False)
    fig.update_yaxes(title_text="Temperature", secondary_y=True)

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=20, b=20)
    )

    return fig
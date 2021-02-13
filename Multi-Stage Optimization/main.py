import cvxpy as cp
import numpy as np
import sys



## Load data
Xi = np.loadtxt('Xi.dat', delimiter='\t')
P  = np.loadtxt('P.dat', delimiter='\t')
p  = np.loadtxt('p.dat', delimiter='\t')
h  = np.loadtxt('h.dat', delimiter='\t')

N,T = Xi.shape


# for a scenario s and time period t says which index should be use
def scenario_variables(s,t):
    i = T
    n = s + N**T - 2
    while i >t:
        n = int(np.ceil(n / 2)-1)
        i = i-1
    return n
sv = scenario_variables

# for a scenario s and time period t says which index should be use

def scenario_parameter(s,t):
    i = T
    n = s + N**T - 1
    while i >t:
        n = int(np.floor(n / N))
        i = i-1
    return n % N, t-1

sp = scenario_parameter

# Calculates the probability of a given scenario
def scenario_prob(s):
    return np.prod([P[sp(s,t)] for t in range(1,T+1)])

prob = scenario_prob

## says at which time variable i corresponds
def time_for_variable(i):
    t = 0
    if i ==0:
        return t
    i = i +1
    while i > 0:
        i = i-N**t
        t = t+1
    return t-1

## Total number of scenarios
scenarios = range(1, N**T +1)

## Variables
x = cp.Variable(N**(T+1) - 1)
y = cp.Variable(N**(T+1) - 1)
## Note that the approach implemented does not require the
## nonanticipativity constraints since it's the same variable
## for the different scenarios.


## Nonnegtivity constraints
constraints = [y >= 0, x>=0]
constraints.append(y[0] == 0)

## Flow constraint, ie what I produce + what I have in inventory
## should equal demand + inventory for next period
for s in scenarios:
    for t in range(1, T+1):
        constraints.append(x[sv(s,t)] + y[sv(s,t-1)]== Xi[sp(s,t)] + y[sv(s,t)])


## Objective: for all scenarios, for all time periods add cost
objective = cp.Minimize(cp.sum([prob(s)* (x[sv(s,t)] * p[t-1] + y[sv(s,t)] * h[t-1]) for t in range(1,T+1) for s in scenarios]))

## Define linear problem and solve
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.GLPK)

## Save results
og_out = sys.stdout

with open('results.txt', 'w') as f:
    sys.stdout = f
    print('Optimal objective value: %f' % result)
    for s in scenarios:
        print()
        print()
        print('Optimal decisions for scenario %i:' %s, ' '.join(['ξ_%i=%.2f' %(t, Xi[sp(s,t)]) for t in range(1,T+1)]))
        print(' '.join(['x_%i=%.2f' %(t, x[sv(s,t)].value) for t in range(0,T+1)]))
        print(' '.join(['y_%i=%.2f' %(t, y[sv(s,t)].value) for t in range(0,T+1)]))





# The following libraries are only required to plot the decision tree.
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
########################



G = nx.DiGraph()

for i in range(N**(T+1)-1):
    node = i #"x_%i" %i + "=%i" % x[i].value + "\n y_%i" %i + "=%i" % y[i].value
    G.add_node(node, label = 'x=%.2f' %x[i].value + '\n y=%.2f' %y[i].value)

for i in range(int(np.floor((N**(T+1)-1)/N))):
    f = i#"x_%i" %i + "=%i" % x[i].value + "\n y_%i" %i + "=%i" % y[i].value
    t = int(np.ceil(i / N))
    for c in range(N):
        j = 2*i+1+c
        s = j#"x_%i" %j + "=%f" % x[j].value + "\n y_%i" %j + "=%f" % y[j].value
        G.add_edge(f, s, label='ξ=%.2f' % Xi[c,time_for_variable(i)])




plt.figure(figsize=(20, 20))
plt.title('Optimal Objective Value %.2f' % result)
pos=graphviz_layout(G, prog='dot')
labels = nx.get_node_attributes(G, 'label')
elabels = nx.get_edge_attributes(G, 'label')
nx.draw(G, pos, labels=labels, arrows=True, node_size=4e3)
nx.draw_networkx_edge_labels(G, pos, labels = elabels)
plt.savefig('optim_result.png')

from math import exp
import math
from random import seed
from random import random
import random

n = 2
p = 3
m = 2
n_epochs = 1000
lr = 0.4
af_name = 'binary_sigmoid'
gates = ['AND', 'OR', 'NAND', 'NOR', 'XOR']
encoding = ['Binary', 'Bipolar']

sigma = 2

while True:
    try:
        gate = int(input('\nEnter number of Gate (AND=1, OR=2, NAND=3, NOR=4, XOR=5): '))
        assert 1 <= gate <= 5
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 5.")
    else:
        print('You choose', gates[gate-1], 'gate.')
        break

# while True:
#     try:
#         n = int(input('\nEnter number of input neurons: '))
#         assert 1 <= n 
#     except ValueError:
#         print("Not an integer! Please enter an integer.")
#     except AssertionError:
#         print("Please enter an integer between greater that 1.")
#     else:
#         print('number of input neurons =', n)
#         break

n=2   
m=1
print('\nNumber of input neurons is 2 and number of output neurons is 1.')
while True:
    try:
        p = int(input('\nEnter number of hidden neurons: '))
        assert 1 <= p  
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between greater that 1.")
    else:
        print('number of hidden neurons =', p)
        break

# while True:
#     try:
#         m = int(input('\nEnter number of output neurons: '))
#         assert 1 <= p  
#     except ValueError:
#         print("Not an integer! Please enter an integer.")
#     except AssertionError:
#         print("Please enter an integer between greater that 1.")
#     else:
#         print('number of output neurons =', m)
#         break

while True:
    try:
        af_name = int(input('\nEnter number of activation function(binary sigmoid=1, bipolar sigmoid=2, sloped sigmoid=3, log=4): '))
        assert 1 <= af_name <= 4
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 4.")
    else:
        if af_name==1:
            af_name='binary_sigmoid'
        elif af_name==2:
            af_name='bipolar_sigmoid'
        elif af_name==3:
            af_name='sloped_sigmoid'
            while True:
                try:
                    sigma = float(input('Enter slope of x: '))
                    assert 0 < sigma  
                except AssertionError:
                    print("Please enter an positive number.")
                else:
                    print('sigma =', sigma)
                    break
        elif af_name==4:
            af_name='log'
        print('You choose', af_name, 'activation function.')
        break

while True:
    try:
        encode = int(input('\nEnter number of encoding inputs(Binary=1, Bipolar=2): '))
        assert 1 <= encode <= 2
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 2.")
    else:
        print('You choose', encoding[encode-1], 'encoding.')
        break
        

while True:
    try:
        n_epochs = int(input('\nEnter number of epochs: '))
        assert n_epochs >= 1
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer greater than 1.")
    else:
        print('Number of epochs is: ', n_epochs)
        break

while True:
    try:
        lr = float(input('\nEnter learning rate: '))
        assert lr >= 0
    except AssertionError:
        print("Please enter an positive number.")
    else:
        print('your learning rate is: ', lr)
        break

while True:
    try:
        weight_type = int(input('\nwhat kind of initialization do you want for weights?(Zero=1, Nguyen-Widrow=2, Gaussian=3, uniform=4): '))
        assert 1 <= weight_type <= 4
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 2.")
    else:
        if weight_type==1:
            V = [[0 for _ in range(n+1)] for _ in range(p)]
            W = [[0 for _ in range(p+1)] for _ in range(m)]
        elif weight_type==2:
            Beta = 0.7*(p**(1/n))
            V = [[random.uniform(-0.5, 0.5) for _ in range(n+1)] for _ in range(p)]
            W = [[random.uniform(-0.5, 0.5) for _ in range(p+1)] for _ in range(m)]
            v_j_abs = []
            for v_j in V:
                v_abs = 0
                for v in v_j:
                    v_abs += v**2
                v_j_abs.append(v_abs**0.5)
            for i in range(len(V)):
                for j in range(len(V[0])):
                    V[i][j] = Beta * V[i][j] / v_j_abs[i]
        elif weight_type==3:
            while True:
                try:
                    mean = float(input('Enter mean: '))
                except AssertionError:
                    print("Please enter a number.")
                else:
                    print('mean: ', mean)
                    break
            while True:
                try:
                    var = float(input('Enter variance: '))
                except AssertionError:
                    print("Please enter a number.")
                else:
                    print('variance: ', var)
                    break
            V = [[random.gauss(mean, var) for _ in range(n+1)] for _ in range(p)]
            W = [[random.gauss(mean, var) for _ in range(p+1)] for _ in range(m)]
        elif weight_type==4:
            while True:
                try:
                    range_ = float(input('Enter range of weights: '))
                except AssertionError:
                    print("Please enter a positive number.")
                else:
                    print(f'range: [{-range_}, {range_}]')
                    break
            V = [[random.uniform(-range_, range_) for _ in range(n+1)] for _ in range(p)]
            W = [[random.uniform(-range_, range_) for _ in range(p+1)] for _ in range(m)]
        break
print('wieghts between input neurons and hidden neurons: ')
for i, vvv in enumerate(V):
    print(f'v_i{i+1}: ' , vvv)
print('wieghts between hidden neurons and output neurons: ')
for i, www in enumerate(W):
    print(f'w_j{i+1}: ' , www)


while True:
    try:
        display = int(input('\nDo you want display loss per epoch?(Yes=1, No=2) (Note: if Yes You should install matplotlib pakage): '))
        assert 1 <= display <= 2
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 2.")
    else:
        break




# dataset_gates = [AND, OR, NAND, NOR, XOR]
dataset_gates = [
    [[0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]],

    [[0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]],

    [[0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]],

    [[0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0]],

    [[0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]],
]

dataset = dataset_gates[gate-1]

if encoding[encode-1] == 'Bipolar':
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == 0:
                dataset[i][j] = -1
                
print('\ndataset:', dataset)



# V = [[random.gauss(0, 1) for _ in range(n+1)] for _ in range(p)]
# W = [[random.gauss(0, 1) for _ in range(p+1)] for _ in range(m)]

activation_function = {
    'relu': lambda x: max(0, x),
    'binary_sigmoid': lambda x: 1 / (1+exp(-x)),
	'bipolar_sigmoid': lambda x: (1-exp(-x)) / (1+exp(-x)),
	'sloped_sigmoid': lambda x: 1 / (1+exp(-sigma*x)),
	'log': lambda x: math.log(1+x) if x>=0 else -math.log(1-x)
}

activation_function_derivative = {
    'relu': lambda x: 1 if x>0 else 0,
    'binary_sigmoid': lambda x: f(x) * (1-f(x)),
	'bipolar_sigmoid': lambda x: 0.5 * (1-f(x)) * (1+f(x)),
	'sloped_sigmoid': lambda x: sigma * f(x) * (1-f(x)),
	'log': lambda x: (1/(1+x)) if x>=0 else (1/(1-x))
}

f = activation_function[af_name]
f_derivative = activation_function_derivative[af_name]


def activate(weights, x):
    activation_in = weights[-1]
    for i in range(len(weights)-1):
        activation_in += weights[i] * x[i]
    return activation_in, f(activation_in)

def feed_forward(V, W, x):
    Z = []
    Z_in = []
    for v in V:
        z_in, z = activate(v, x)
        Z.append(z)
        Z_in.append(z_in)

    Y = []
    Y_in = []
    for w in W:
        y_in, y = activate(w, Z)
        Y.append(y)
        Y_in.append(y_in)
    return Y, Z, Y_in, Z_in


def predict(V, W, x):
    o, _, _, _ = feed_forward(V, W, x)
    return o
    

def train(n_epochs, V, W):
    loss_per_epoch = []
    for epoch in range(n_epochs):
        loss = 0
        for x_ in X:
            x = x_[:-m]
            t = x_[-m:]
            if not isinstance(t, list):
                t = [t]
            Y, Z, Y_in, Z_in = feed_forward(V, W, x)

            for (target, y_predicted) in zip(t, Y):
                loss += (target - y_predicted)**2
            

            delta_k = [0 for _ in range(len(Y))]
            for k in range(len(Y)):
                delta_k[k] = (t[k]-Y[k]) * f_derivative(Y_in[k])

            Delta_w_jk = [[0 for _ in range(len(Z)+1)] for _ in range(len(Y))]
            for k in range(len(Y)):
                Delta_w_jk[k][-1] = lr * delta_k[k]
                for j in range(len(Z)):
                    Delta_w_jk[k][j] = lr * delta_k[k] * Z[j]
            
            delta_inj = []
            for j in range(p):
                d = 0
                for k in range(len(delta_k)):
                    d += delta_k[k] * W[k][j]
                delta_inj.append(d)

            delta_j = []
            for j in range(p):
                delta_j.append(delta_inj[j] * f_derivative(Z_in[j]))

            Delta_v_ij = [[0 for _ in range(len(x)+1)] for _ in range(len(Z))]
            for j in range(len(Z)):
                Delta_v_ij[j][-1] = lr * delta_j[j]
                for i in range(len(x)):
                    Delta_v_ij[j][i] = lr * delta_j[j] * x[i]

            W_new = [[] for _ in range(len(Y))]
            for k in range(len(Y)):
                for j in range(len(Z)):
                    W_new[k].append(W[k][j]+Delta_w_jk[k][j])
                W_new[k].append(W[k][-1]+Delta_w_jk[k][-1])

            V_new = [[] for _ in range(len(Z))]
            for j in range(len(Z)):
                for i in range(len(x)):
                    V_new[j].append(V[j][i]+Delta_v_ij[j][i])
                V_new[j].append(V[j][-1]+Delta_v_ij[j][-1])

            W = W_new
            V = V_new
        loss_per_epoch.append(loss)
        print(f'epoch = {epoch+1:05}, loss = {loss:.16f}')
    return V, W, loss_per_epoch

X = dataset

V, W, loss_per_epoch = train(n_epochs, V, W)

print('\nEvaluation Model:')
for x_ in X:
    x = x_[:-m]
    o = predict(V, W, x)
    print('input:', x, '--> target:', x_[-1], ', output:', o)

if display==1:
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, n_epochs+1)], loss_per_epoch, linewidth=3)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid(True)
    plt.show()

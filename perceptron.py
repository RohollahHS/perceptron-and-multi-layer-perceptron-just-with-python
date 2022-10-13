from math import exp
gates = ['AND', 'OR', 'NAND', 'NOR', 'XOR']
encoding = ['Binary', 'Bipolar']
rule = ['Hebbian', 'Delta', 'Gradient Descent']
learning_type = ['Online', 'Batch']
e = 2.71828182

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
    
while True:
    try:
        n_rule = int(input('\nEnter number of Rule learning (Hebbian=1, Delta=2, Gradient Descent=3): '))
        assert 1 <= n_rule <= 3
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 3.")
    else:
        print('You choose', rule[n_rule-1], 'rule.')
        break

while True:
    try:
        type_ = int(input('\nEnter type of learning (Online Learninig=1, Batch Learning=2): '))
        assert 1 <= type_ <= 2
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 2.")
    else:
        print('You choose', learning_type[type_-1], 'Learning.')
        break

while True:
    try:
        learning_rate = float(input('\nEnter learning rate (between 0 and 1): '))
        assert 0 < learning_rate < 1
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter a number between 0 and 1.")
    else:
        print('your learning rate is: ', learning_rate)
        break



# w = input('\nEnter your initial weight (Note: this number repeats for other weights): ')
# weights = [float(w) for _ in range(3)]
import random
weights=[]
# while True:
#     try:
#         w = float(input('\nEnter your initial weights (between 0 and 1): '))
#         assert 0 <= w <= 1
#     except ValueError:
#         print("Not an integer! Please enter an integer.")
#     except AssertionError:
#         print("Please enter a number between 0 and 1.")
#     else:
#         weights = [w for _ in range(3)]
#         print('your initial weights is: ', weights)
#         break

while True:
    try:
        weight_type = int(input('\nwhat kind of initialization do you want for weights?(Zero=1, Gaussian=2, uniform=3): '))
        assert 1 <= weight_type <= 3
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer between 1 and 2.")
    else:
        if weight_type==1:
            weights = [0 for _ in range(3)]
        elif weight_type==2:
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
            weights = [random.gauss(mean, var) for _ in range(3)]
        elif weight_type==3:
            while True:
                try:
                    range_ = float(input('Enter range of weights: '))
                except AssertionError:
                    print("Please enter a positive number.")
                else:
                    print(f'range: [{-range_}, {range_}]')
                    break
            weights = [random.uniform(-range_, range_) for _ in range(3)]
        break
print('your initial weights is: ', weights)

while True:
    try:
        iteration = int(input('\nEnter number of iteration: '))
        assert iteration >= 1
    except ValueError:
        print("Not an integer! Please enter an integer.")
    except AssertionError:
        print("Please enter an integer greater than 1")
    else:
        print('Number of iteration: ', iteration)
        break

if n_rule == 3 or n_rule==2:
    encode = 1
else:
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


# Hebbia rule with Step activation function ================================ 
if rule[n_rule-1] == 'Hebbian':
    def predict(sample, weights):
        activation = weights[0]
        for i in range(len(sample)-1):
            activation += sample[i] * weights[i+1]

        if encoding[encode-1] == 'Binary':
            prediction = 1 if activation > 0 else 0
        else:
            prediction = 1 if activation > 0 else -1
        target = sample[-1]

        return prediction, target


    def learning(sample, weights, learning_rate=0.1):
        prediction, target = predict(sample, weights)
        print('\nsample', sample)
        print(f'target={target}, prediction={prediction}')
        print('old weights:', weights)
        # delta_w0 = learning_rate * (target - prediction)
        delta_w0 = learning_rate * (target - prediction)
        weights[0] = weights[0] + delta_w0 # wi_new = wi_old + delta_w0
        for i in range(len(sample)-1):
            delta_wi = learning_rate * (target - prediction) * sample[i] # smpale[i] == xi
            weights[i+1] = weights[i+1] + delta_wi # wi_new = wi_old + delta_wi
        print('new weights:', weights)
        return weights

    def batch_learning(dataset, weights, learning_rate=0.1):
        delta_w0 = 0
        delta_wi_ = [0 for _ in range(3)]
        for sample in dataset:
            prediction, target = predict(sample, weights)
            print('\nsample', sample)
            print(f'target={target}, prediction={prediction}')
            delta_wi_[0] += learning_rate * (target - prediction)
            for i in range(len(sample)-1):
                delta_wi_[i+1] += learning_rate * (target - prediction) * sample[i] # smpale[i] == xi
        print('\nold weights:', weights)
        weights[0] = weights[0] + delta_wi_[0]
        for i in range(len(sample)-1):
            weights[i+1] = weights[i+1] + delta_wi_[i+1]
        print('new weights:', weights)
        return weights


# Delta rule with Linear activation function ===========================
if rule[n_rule-1] == 'Delta':
    def predict(sample, weights):
        activation = weights[0]
        for i in range(len(sample)-1):
            activation += sample[i] * weights[i+1]

        prediction = activation
        target = sample[-1]

        return prediction, target


    def learning(sample, weights, learning_rate=0.1):
        prediction, target = predict(sample, weights)
        print('\nsample', sample)
        print(f'target={target}, prediction={prediction}')
        print('old weights:', weights)
        delta_w0 = learning_rate * (target - prediction)
        weights[0] = weights[0] + delta_w0 # wi_new = wi_old + delta_w0
        for i in range(len(sample)-1):
            delta_wi = learning_rate * (target - prediction) * sample[i] # smpale[i] == xi
            weights[i+1] = weights[i+1] + delta_wi # wi_new = wi_old + delta_wi
        print('new weights:', weights)
        return weights

    def batch_learning(dataset, weights, learning_rate=0.1):
        delta_w0 = 0
        delta_wi_ = [0 for _ in range(3)]
        for sample in dataset:
            prediction, target = predict(sample, weights)
            print('\nsample', sample)
            print(f'target={target}, prediction={prediction}')
            delta_wi_[0] += learning_rate * (target - prediction)
            for i in range(len(sample)-1):
                delta_wi_[i+1] += learning_rate * (target - prediction) * sample[i] # smpale[i] == xi
        print('\nold weights:', weights)
        weights[0] = weights[0] + delta_wi_[0]
        for i in range(len(sample)-1):
            weights[i+1] = weights[i+1] + delta_wi_[i+1]
        print('new weights:', weights)
        return weights

# Gradient Descent with Sigmoid activation fuction  ============
if rule[n_rule-1] == 'Gradient Descent':
    def sigmoid(x):
        if encoding[encode-1] == 'Binary':
            return 1 / (1 + e**(-x))
        else: 
            return (1 - e**(-x)) / (1 + e**(-x))

    def predict(sample, weights):

        activation = weights[0]
        for i in range(len(sample)-1):
            activation += sample[i] * weights[i+1]
        prediction = sigmoid(activation)    
        target = sample[-1]
        return prediction, target

    def learning(sample, weights, learning_rate=0.1):
        prediction, target = predict(sample, weights)
        print('\nsample', sample)
        print(f'target={target}, prediction={prediction}')
        print('old weights:', weights)
        delta_w0 = learning_rate * (target - prediction) * prediction * (1 - prediction)
        weights[0] = weights[0] + delta_w0 # wi_new = wi_old + delta_w0
        for i in range(len(sample)-1):
            delta_wi = learning_rate * (target - prediction) * prediction * (1 - prediction) * sample[i] # smpale[i] == xi
            weights[i+1] = weights[i+1] + delta_wi # wi_new = wi_old + delta_wi
        print('new weights:', weights)
        return weights

    def batch_learning(dataset, weights, learning_rate=0.1):
        delta_w0 = 0
        delta_wi_ = [0 for _ in range(3)]
        for sample in dataset:
            prediction, target = predict(sample, weights)
            print('\nsample', sample)
            print(f'target={target}, prediction={prediction}')
            delta_wi_[0] += learning_rate * (target - prediction) * prediction * (1-prediction)
            for i in range(len(sample)-1):
                delta_wi_[i+1] += learning_rate * (target - prediction) * sample[i] # smpale[i] == xi
        print('\nold weights:', weights)
        weights[0] = weights[0] + delta_wi_[0]
        for i in range(len(sample)-1):
            weights[i+1] = weights[i+1] + delta_wi_[i+1]
        print('new weights:', weights)
        return weights

def train(dataset, weights, learning_rate, iteration):
    for i in range(iteration):
        print(f'\n----------------------------------------------\niteration {i+1}th:')
        predictions = []
        targets = []

        if learning_type[type_-1] == 'Online':
            for sample in dataset:
                weights = learning(sample, weights, learning_rate)

        if learning_type[type_-1] == 'Batch':
            weights = batch_learning(dataset, weights, learning_rate)

        # evaluation
        for sample in dataset:
            prediction, target = predict(sample, weights)
            predictions.append(prediction)
            targets.append(target)

        loss = 0
        for j in range(len(targets)):
            loss += (targets[j] - predictions[j])**2

        print(f'\nEvaluation model after {i+1}th iteration:')
        print('Predictions:', predictions)
        print('Targets:    ', targets)
        print(f'Loss: {0.5*loss}')
        if predictions == targets:
            print(f'\nModel learned dataset after {i+1} iteration!\n\n')
            break


train(dataset, weights, learning_rate, iteration)

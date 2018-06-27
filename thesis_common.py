import torch.nn.init as init
import torch
from torch.autograd import Variable
import scipy.stats
import numpy as np


def forward(input_state, context_state, w1, w2, activation_function):
    input_with_context = torch.cat((input_state, context_state), 1)
    context_state = activation_function(input_with_context.mm(w1))
    output_state = context_state.mm(w2)
    return output_state, context_state


def train_single(w1, w2, lr, activation_function, training_scenarios, hidden_size):
    total_loss = 0

    for scenario in training_scenarios:
        x = Variable(torch.FloatTensor(scenario[1]), requires_grad=False)
        y = Variable(torch.FloatTensor(scenario[2]), requires_grad=False)
        context_state = Variable(torch.zeros((1, hidden_size)).type(torch.FloatTensor), requires_grad=True)

        for j in range(len(x)):
            input_state = x[j:(j + 1)]
            target = y[j:(j + 1)]
            (pred, new_context_state) = forward(input_state, context_state, w1, w2, activation_function)
            loss = (pred - target).pow(2).sum() / 2
            total_loss += loss
            loss.backward()
            w1.data -= lr * w1.grad.data
            w2.data -= lr * w2.grad.data
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            context_state = Variable(new_context_state.data)

    return total_loss


def train_and_test_at_intervals(w1, w2, epochs, lr, activation_function, test_set, training_scenarios, hidden_size, interval=100):
    epoch = []
    accuracy = []

    for i in range(epochs):
        total_loss = train_single(w1, w2, lr, activation_function, training_scenarios, hidden_size)

        if i % 1000 == 0:
            print("Epoch: {} loss {}".format(i, total_loss.data[0]))

        if i % interval == 0:
            epoch.append(i)
            accuracy.append(test(w1, w2, activation_function, test_set, hidden_size))

    return epoch, accuracy


def train(w1, w2, epochs, lr, activation_function, training_scenarios, hidden_size):
    for i in range(epochs):
        total_loss = train_single(w1, w2, lr, activation_function, training_scenarios, hidden_size)

        if i % 1000 == 0:
            print("Epoch: {} loss {}".format(i, total_loss.data[0]))


def train_and_test_at_intervals_clone_weights(w1, w2, epochs, lr, activation_function, test_set, training_scenarios, hidden_size, interval=100):
    weights1 = Variable(w1.clone(), requires_grad=True)
    weights2 = Variable(w2.clone(), requires_grad=True)

    return train_and_test_at_intervals(weights1, weights2, epochs, lr, activation_function, test_set, training_scenarios, hidden_size, interval)


def loss_measure(predicted, actual):
    return (actual - predicted) ** 2


def test(w1, w2, activation_function, test_set, hidden_size):
    losses = []

    for scenario in test_set:
        context_state = Variable(torch.zeros((1, hidden_size)).type(torch.FloatTensor), requires_grad=False)
        x = Variable(torch.FloatTensor(scenario[1]), requires_grad=False)
        y = scenario[2]
        loss = 0

        for i in range(len(x)):
            inp = x[i:i+1]
            (pred, new_context_state) = forward(inp, context_state, w1, w2, activation_function)
            context_state = new_context_state
            loss += loss_measure(pred.data.numpy().ravel()[0], y[i][0])

        losses.append(loss / len(x))

    return sum(losses) / float(len(losses))


def initialize_random_subject(input_size, hidden_size, output_size):
    w1 = torch.FloatTensor(input_size, hidden_size).type(torch.FloatTensor)
    init.normal(w1, 0.0, 0.4)
    w2 = torch.FloatTensor(hidden_size, output_size).type(torch.FloatTensor)
    init.normal(w2, 0.0, 0.3)

    return w1, w2


def get_mean_and_sems(subjects):
    means = []
    sems = []

    for i in range(len(subjects[0][0])):
        means.append(np.mean([y[i] for (x, y) in subjects]))
        sems.append(scipy.stats.sem([y[i] for (x, y) in subjects]))

    return means, sems


#def get_mean_and_stds(subjects):
#    means = []
#    stds = []
#
#    for i in range(len(subjects[0][0])):
#        means.append(np.mean([y[i] for (x, y) in subjects]))
#        stds.append(np.std([y[i] for (x, y) in subjects]))
#
#    return (means, stds)
import matplotlib.pyplot as plt
import csv
import numpy as np
import math

b0 = 0
b1 = 0
b2 = 0
w_initial = np.random.normal(0, 1, (2, 1))
u_initial = np.random.normal(0, 1, (2, 1))
v_initial = np.random.normal(0, 1, (2, 1))
learning_rate = 0.1
num_of_epochs = 9000
w = []
u = []
v = []
for i in range(2):
    w.append(w_initial[i][0])
    u.append(u_initial[i][0])
    v.append(v_initial[i][0])

# reading the inputs
file_location = "location to dataset.csv"
points = []
labels = []
with open(file_location, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        points.append(row)
for i in range(1, len(points)):
    for j in range(3):
        points[i][j] = float(points[i][j])
points.remove(points[0])

# drawing
for j in range(len(points)):
    if points[j][2] == 0:
        plt.scatter(points[j][0], points[j][1], color='red')
    else:
        plt.scatter(points[j][0], points[j][1], color='blue')
plt.show()

# figuring out training and test sets
np.random.shuffle(points)
training_set, test_set = points[0:int(len(points) * 3 / 4)], points[int(len(points) * 3 / 4):]
training_set = np.array(training_set)
test_set = np.array(test_set)
training_set_xs = training_set[:, 0:2]
training_set_labels = training_set[:, 2]
test_set_xs = test_set[:, 0:2]
test_set_labels = test_set[:, 2]

# drawing
for j in range(len(training_set)):
    if training_set[j][2] == 0:
        plt.scatter(training_set[j][0], training_set[j][1], color='red')
    else:
        plt.scatter(training_set[j][0], training_set[j][1], color='blue')
plt.savefig("trainingset_none_linear.jpg")
plt.show()
# drawing
for j in range(len(test_set)):
    if test_set[j][2] == 0:
        plt.scatter(test_set[j][0], test_set[j][1], color='red')
    else:
        plt.scatter(test_set[j][0], test_set[j][1], color='blue')
plt.savefig("testset_none_linear.jpg")
plt.show()


def sigmoid(x):
    if x > 700:
        return 1
    elif x < -700:
        return 0
    return 1 / (math.exp(-x) + 1)


def compute_sigmoid(inp, weight, bias):
    return sigmoid(inp[0] * weight[0] + inp[1] * weight[1] + bias)


for epoch in range(num_of_epochs):
    all_as = []
    error = 0
    grad = {"b0": 0, "b1": 0, "w0": 0, "w1": 0, "u0": 0, "u1": 0, "b2": 0, "v0": 0, "v1": 0}
    for i in range(len(training_set_xs)):
        a = []
        first_layer_a0_sigmoid = compute_sigmoid(training_set_xs[i], w, b0)
        a.append(first_layer_a0_sigmoid)
        first_layer_a1_sigmoid = compute_sigmoid(training_set_xs[i], u, b1)
        a.append(first_layer_a1_sigmoid)
        all_as.append(a)
        second_layer_a2_sigmoid = compute_sigmoid(a, v, b2)
        error += ((training_set_labels[i] - second_layer_a2_sigmoid) ** 2)
        grad["b2"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                    second_layer_a2_sigmoid - training_set_labels[i])
        grad["v0"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                    second_layer_a2_sigmoid - training_set_labels[i]) * first_layer_a0_sigmoid
        grad["v1"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                    second_layer_a2_sigmoid - training_set_labels[i]) * first_layer_a1_sigmoid
        grad["w0"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                    second_layer_a2_sigmoid - training_set_labels[i]) * v[0] * first_layer_a0_sigmoid * (
                                  1 - first_layer_a0_sigmoid) * training_set_xs[i][0]
        grad["u0"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                second_layer_a2_sigmoid - training_set_labels[i]) * v[1] * first_layer_a1_sigmoid * (
                              1 - first_layer_a1_sigmoid) * training_set_xs[i][0]
        grad["w1"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                second_layer_a2_sigmoid - training_set_labels[i]) * v[0] * first_layer_a0_sigmoid * (
                              1 - first_layer_a0_sigmoid) * training_set_xs[i][1]
        grad["u1"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                second_layer_a2_sigmoid - training_set_labels[i]) * v[1] * first_layer_a1_sigmoid * (
                              1 - first_layer_a1_sigmoid) * training_set_xs[i][1]
        grad["b0"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                second_layer_a2_sigmoid - training_set_labels[i]) * v[0] * first_layer_a0_sigmoid * (
                              1 - first_layer_a0_sigmoid)
        grad["b1"] += second_layer_a2_sigmoid * (1 - second_layer_a2_sigmoid) * (
                second_layer_a2_sigmoid - training_set_labels[i]) * v[1] * first_layer_a1_sigmoid * (
                              1 - first_layer_a1_sigmoid)
    # updating the parameters
    b2 = b2 - (learning_rate * grad["b2"] / len(training_set_xs))
    b1 = b1 - (learning_rate * grad["b1"] / len(training_set_xs))
    b0 = b0 - (learning_rate * grad["b0"] / len(training_set_xs))
    v[1] = v[1] - (learning_rate * grad["v1"] / len(training_set_xs))
    v[0] = v[0] - (learning_rate * grad["v0"] / len(training_set_xs))
    w[1] = w[1] - (learning_rate * grad["w1"] / len(training_set_xs))
    w[0] = w[0] - (learning_rate * grad["w0"] / len(training_set_xs))
    u[1] = u[1] - (learning_rate * grad["u1"] / len(training_set_xs))
    u[0] = u[0] - (learning_rate * grad["u0"] / len(training_set_xs))
    J = error / len(training_set)
    print(J)
# ress shows the prediction of the model for the test data
ress = []
for i in range(len(test_set)):
    a = []
    first_layer_a0_sigmoid = compute_sigmoid(test_set_xs[i], w, b0)
    a.append(first_layer_a0_sigmoid)
    first_layer_a1_sigmoid = compute_sigmoid(test_set_xs[i], u, b1)
    a.append(first_layer_a1_sigmoid)
    second_layer_a2_sigmoid = compute_sigmoid(a, v, b2)
    ress.append(round(second_layer_a2_sigmoid))
# sum is the number of mistakes the model makes in its predictions
summ = 0
for index in range(len(test_set_labels)):
    summ += math.fabs(ress[index] - test_set_labels[index])
print(summ)
# drawing the result
for j in range(len(test_set_xs)):
    if ress[j] == 0:
        plt.scatter(test_set_xs[j][0], test_set_xs[j][1], color='red')
    else:
        plt.scatter(test_set_xs[j][0], test_set_xs[j][1], color='blue')
plt.savefig("result_none_linear.jpg")
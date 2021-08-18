import matplotlib.pyplot as plt
import csv
import numpy as np
import math

b = 0
w_initial = np.random.normal(0, 1, (2, 1))
learning_rate = 0.1
num_of_epochs = 3500
w = []
for i in range(2):
    w.append(w_initial[i][0])
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

# drawing them
for j in range(len(points)):
    if points[j][2] == 0:
        plt.scatter(points[j][0], points[j][1], color='red')
    else:
        plt.scatter(points[j][0], points[j][1], color='blue')
plt.savefig("allthepoints.jpg")
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

# drawing them
for j in range(len(training_set)):
    if training_set[j][2] == 0:
        plt.scatter(training_set[j][0], training_set[j][1], color='red')
    else:
        plt.scatter(training_set[j][0], training_set[j][1], color='blue')
plt.savefig("trainingset_linear.jpg")
plt.show()
# drawing them
for j in range(len(test_set)):
    if test_set[j][2] == 0:
        plt.scatter(test_set[j][0], test_set[j][1], color='red')
    else:
        plt.scatter(test_set[j][0], test_set[j][1], color='blue')
plt.savefig("testset_linear.jpg")
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
    error = 0
    grad = {"b": 0, "w0": 0, "w1": 0}
    for i in range(len(training_set_xs)):
        sigmoid_result = compute_sigmoid(training_set_xs[i], w, b)
        grad["b"] += sigmoid_result * (1 - sigmoid_result) * (
                sigmoid_result - training_set_labels[i])
        grad["w0"] += sigmoid_result * (1 - sigmoid_result) * (
                sigmoid_result - training_set_labels[i]) * training_set_xs[i][0]
        grad["w1"] += sigmoid_result * (1 - sigmoid_result) * (
                sigmoid_result - training_set_labels[i]) * training_set_xs[i][1]
        error += ((training_set_labels[i] - sigmoid_result) ** 2)
    # updating the parameters
    b = b - (learning_rate * grad["b"] / len(training_set_xs))
    w[0] = w[0] - (learning_rate * grad["w0"] / len(training_set))
    w[1] = w[1] - (learning_rate * grad["w1"] / len(training_set))
    J = error / len(training_set)
    print(J)
# ress shows the prediction of the model for the test data
ress = []
for i in range(len(test_set)):
    ress.append(round(compute_sigmoid(test_set[i], w, b)))
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
plt.savefig("result_linear.jpg")
plt.show()
import numpy as np
import gd

n = 100
x = np.linspace(-5, 5, n)
noise = np.random.normal(0, 0.01, n)
y = np.sin(x) + noise

p = np.random.permutation(n)
train = p[:70]
test = p[70:]

x_train = x[train].reshape(-1, 1)
y_train = y[train]
x_test = x[test].reshape(-1, 1)
y_test = y[test]

descend_names = ["full", "stochastic", "momentum", "adam"]

for descend_name in descend_names:
    descend = gd.get_descent({"descend_name": descend_name,
                              "regularized": False,
                              "kwargs": {"dimension": 1}})

    k = 10_000
    for t in range(k):
        descend.step(x_train, y_train)

    loss = descend.calc_loss(x_test, y_test)
    print(f"{descend_name}: weights = {descend.w}, test_loss = {loss}")
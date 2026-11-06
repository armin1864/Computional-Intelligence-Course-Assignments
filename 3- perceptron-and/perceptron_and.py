import numpy as np


x = np.array([
[0, 0],
[0, 1],
[1, 0],
[1, 1]
])


target = np.array([-1, -1, -1, 1])

a = 1
theta = 0.2


w = np.zeros(2)
b = 0


def activation(net):
    if net >= theta:
        return 1
    return -1


def train():
    global w, b

    epoch = 0
    while True:
        errors = 0
        epoch += 1
        for i in range(len(x)):
            net = np.dot(x[i], w) + b
            y = activation(net)
            if y != target[i]:
                w += a * target[i] * x[i]
                b += a * target[i]
                errors += 1
        print(f"Epoch {epoch}: W={w}, Bias={b}, Errors={errors}")
        if errors == 0:
            break
    print("Final weights:", w)
    print("Final bias:", b)


def predict(x1, x2):
    x = np.array([x1, x2])
    net = np.dot(x, w) + b
    return activation(net)



train()

print("\n— User Input —")

x1 = int(input("Enter X1 (0 or 1): "))
x2 = int(input("Enter X2 (0 or 1): "))

if x1 not in [0, 1] or x2 not in [0, 1]:
    print("Invalid input! Please enter 0 or 1.")
else:
    result = predict(x1, x2)

if result == 1:
    print("AND Output: 1")
else:
    print("AND Output: 0")
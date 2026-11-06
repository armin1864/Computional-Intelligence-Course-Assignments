# simple hebb network implemented for AND gate
# im using matplotlib for drawing charts in terminal 
import matplotlib.pyplot as plt

# training samples are 4 states of AND gate
train_samples = [
    (-1, -1, -1),
    (-1, 1, -1),
    (1, -1, -1),
    (1, 1, 1)
]

# weights and bias are 0 in the start
w1=0
w2=0
b=0

#we are using bipolar y^ function in this example
def y_out(net):
    if net >=0:
        return 1
    return -1


def compute_net(x1, x2):
    net = w1*x1 + w2*x2 + b
    return net


#training in hebb network:
# w1_new= w1_old + x1*t
# w2_new = w2_old + x2*t
# b_new = b_old + t
def update_prams(x1, x2, t):
    global w1, w2, b
    w1 += x1*t
    w2 += x2*t
    b += t

def train():
    global train_samples
    for x1, x2, t in train_samples:
        update_prams(x1, x2, t)
        net = compute_net(x1, x2)
        out = y_out(net)
        print("\n")
        print(f"Input: ({x1}, {x2})")
        print(f"Target: {t}")
        print(f"Output: {out}")
        print(f"Weights: w1={w1}, w2={w2}")
        print(f"Bias: b={b}")
        draw_decision_boundary(w1, w2, b, x1, x2, t)



# matplotlib
def draw_decision_boundary(w1, w2, b, x1, x2, target):
    plt.clf()

    for a, c, t in train_samples:
        if t == 1:
            plt.scatter(a, c, color="green", marker="o",
                        s=100, label="Target +1")
        else:
            plt.scatter(a, c, color="red", marker="x",
                        s=100, label="Target -1")
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    # decision line: w1*x1 + w2*x2 + b = 0
    if w1 != 0 or w2 != 0:
        if w2 != 0:
            x_values = [-2, 2]
            y_values = [
                -(w1 * x + b) / w2
                for x in x_values
            ]
            plt.plot(x_values, y_values, "b-", label="Decision boundary")
        else:
            x_position = -b / w1
            plt.axvline(x=x_position, color="blue",
                        label="Decision boundary")
    else:
        plt.text(
            0, 0,
            "No defined decision boundary yet",
            ha="center"
        )
    plt.title(
        f"Input=({x1}, {x2}), Target={target}\n"
        f"w1={w1}, w2={w2}, b={b}"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.pause(1)

plt.ion()
plt.figure(figsize=(7, 6))
print("hebb AND")
print("-" * 40)


train()

# user input
while True:
    try:
        print("\nEnter bipolar input values (-1 or +1).")
        user_x1 = int(input("Enter X1: "))
        user_x2 = int(input("Enter X2: "))
    except ValueError:
        print("Invalid input. Please enter -1 or +1.")
        continue

    if user_x1 not in (-1, 1) or user_x2 not in (-1, 1):
        print("Invalid input. Only -1 and +1 are allowed.")
        continue

    net = compute_net(user_x1, user_x2)
    user_out = y_out(net)

    print("Network output:", user_out)
    print("Expected AND result:", 1 if user_x1 == 1 and user_x2 == 1 else -1)

    # Show the user input on the existing plot.
    plt.scatter(user_x1, user_x2, color="orange", marker="*", s=220,
                label="User input")
    plt.title(
        f"Trained Bipolar AND\n"
        f"User input=({user_x1}, {user_x2}), Network output={user_out}"
    )
    plt.legend()
    plt.draw()
    plt.pause(0.1)

    again = input("Try another input? (y/n): ").strip().lower()
    if again != "y":
        break
    

plt.ioff()
plt.show()
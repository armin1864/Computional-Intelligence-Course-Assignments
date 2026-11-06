import tkinter as tk
from tkinter import messagebox

from samples import X_SAMPLES, O_SAMPLES


def convert_sample(sample):
    if len(sample) != 64:
        raise ValueError("Each sample must contain exactly 64 characters.")

    return [1 if pixel != "." else -1 for pixel in sample]


x_samples = [convert_sample(sample) for sample in X_SAMPLES]
o_samples = [convert_sample(sample) for sample in O_SAMPLES]

a = 1
theta = 0.2
MAX_EPOCHS = 1000


weights = [0] * 64
bias = 0


def activation(net):
    if net >= theta:
        return 1
    return -1


def predict(pattern):

    net = sum(
        weights[i] * pattern[i]
        for i in range(64)
    ) + bias
    return activation(net)



def train_network():

    #we want to get y=1 for x and y=-1 for o
    training_data = []
    for pattern in x_samples:
        training_data.append((pattern, 1))

    for pattern in o_samples:
        training_data.append((pattern, -1))

    global weights, bias
    weights = [0] * 64
    bias = 0
    for epoch in range(MAX_EPOCHS):
        errors = 0
        for pattern, target in training_data:
            output = predict(pattern)
            if output != target:
                for i in range(64):
                    weights[i] += a * target * pattern[i]
                bias += a * target
                errors += 1
        if errors == 0:
            print("Training completed!")
            print("Epochs:", epoch + 1)
            return True
    print("Maximum epochs reached!")
    return False


root = tk.Tk()
root.title("Hebbian Learning - X and O")
root.resizable(True, True)

user_pattern = [-1] * 64
cells = []

grid_frame = tk.Frame(root)
grid_frame.pack(padx=15, pady=15)


def toggle_cell(index):
    if user_pattern[index] == -1:
        user_pattern[index] = 1
        cells[index].configure(bg="black")
    else:
        user_pattern[index] = -1
        cells[index].configure(bg="white")


for row in range(8):
    for col in range(8):
        index = row * 8 + col

        cell = tk.Button(
            grid_frame,
            width=3,
            height=3,
            bg="white",
            relief="solid",
            command=lambda i=index: toggle_cell(i)
        )
        cell.grid(row=row, column=col, padx=1, pady=1)
        cells.append(cell)


def update_sample_count():
    sample_label.configure(
        text=f"X samples: {len(x_samples)}     O samples: {len(o_samples)}"
    )


def clear_grid():
    global user_pattern

    user_pattern = [-1] * 64

    for cell in cells:
        cell.configure(bg="white")

    result_label.configure(text="Grid cleared.")


def recognize():
    result = "x" if predict(user_pattern) == 1 else "o"
    result_label.configure(text=f"Prediction: {result}")


def add_sample(character):
    pattern = user_pattern.copy()

    if all(value == -1 for value in pattern):
        messagebox.showwarning("Empty grid")
        return

    if character == "X":
        x_samples.append(pattern)
    else:
        o_samples.append(pattern)

   
    train_network()
    update_sample_count()

    result_label.configure(text=f"New {character} sample added and network retrained.")


button_frame = tk.Frame(root)
button_frame.pack(pady=5)

tk.Button(
    button_frame, text="Recognize", width=14, command=recognize
).grid(row=0, column=0, padx=4, pady=4)

tk.Button(
    button_frame, text="Clear", width=14, command=clear_grid
).grid(row=0, column=1, padx=4, pady=4)

tk.Button(
    button_frame, text="Add as X", width=14,
    command=lambda: add_sample("X")
).grid(row=1, column=0, padx=4, pady=4)

tk.Button(
    button_frame, text="Add as O", width=14,
    command=lambda: add_sample("O")
).grid(row=1, column=1, padx=4, pady=4)

result_label = tk.Label(
    root, text="Draw X or O, then choose an action.", font=("Arial", 12)
)
result_label.pack(pady=8)

sample_label = tk.Label(root, text="", font=("Arial", 10))
sample_label.pack(pady=5)



train_network()
update_sample_count()

root.mainloop()
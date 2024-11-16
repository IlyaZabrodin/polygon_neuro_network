import numpy as np
import matplotlib.pyplot as plt

import utils

images, labels = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 23716))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (2, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((2, 1))

epochs = 5
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Epoch №{epoch}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

        # Forward propagation (to output layer)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        # Loss / Error calculation
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation (output layer)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Backpropagation (hidden layer)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

        # DONE

    # print some debug info between epochs
    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# SAVING
np.save('weights_input_to_hidden.npy', weights_input_to_hidden)
np.save('weights_hidden_to_output.npy', weights_hidden_to_output)
np.save('bias_input_to_hidden.npy', bias_input_to_hidden)
np.save('bias_hidden_to_output.npy', bias_hidden_to_output)

# CHECK CUSTOM
test_image = plt.imread("data/polygon0001.png", format="png")
plt.imshow(test_image)

test_image = 1 - (test_image.astype("float32") / 255)

# Reshape
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1] * test_image.shape[2]))

# Predict
image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

if output.argmax() == 1:
	plt.title(f"NN suggests the CUSTOM number is: Convex")
else:
	plt.title(f"NN suggests the CUSTOM number is: Convac")
plt.show()
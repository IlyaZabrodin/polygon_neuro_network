import numpy as np
import matplotlib.pyplot as plt


weights_input_to_hidden = np.load('weights_input_to_hidden.npy')
weights_hidden_to_output = np.load('weights_hidden_to_output.npy')
bias_input_to_hidden = np.load('bias_input_to_hidden.npy')
bias_hidden_to_output = np.load('bias_hidden_to_output.npy')

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
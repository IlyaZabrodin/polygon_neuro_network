import numpy as np

def load_dataset():
	with np.load("network_dataset.npz", allow_pickle=True) as f:
		# convert from RGB to Unit RGB
		x_train = 1 - (f['x'].astype("float32") / 255)

		# reshape from (100, 77, 77, 4) into (100, 23761)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))

		# labels
		y_train = f['y']

		# convert to output layer format
		y_train = np.eye(2)[y_train]

		return x_train, y_train
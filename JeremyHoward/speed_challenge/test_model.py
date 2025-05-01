from tinygrad import Tensor, nn
import numpy as np

# Instantiate the model
#model = Model()

# Load the saved model
model_save_path = "./models/linear_sgd.npz"
print(f"Loading model from {model_save_path}")

# Load the .npz file
loaded_params = np.load(model_save_path)

# Assign loaded arrays back to the model's parameters
params = nn.state.get_parameters(model)
for i, param in enumerate(params):
    param.assign(Tensor(loaded_params[f'param_{i}']))

print(f"Model loaded successfully from {model_save_path}")

# Now you can use the model for testing
# Example: Forward pass on test data
def forward(X):
    Tensor.training = False  # Disable training mode (e.g., for dropout)
    return model(X)

# Example: Load test data and run inference
X_test = np.load("./data/flow_images.npy", mmap_mode='r')[:100].astype(np.float32) / 255.0  # Example: first 100 frames
X_test_tensor = Tensor(X_test)
predictions = forward(X_test_tensor).numpy()
print(f"Predictions shape: {predictions.shape}")
print(predictions)

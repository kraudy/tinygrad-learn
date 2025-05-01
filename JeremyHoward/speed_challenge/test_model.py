from tinygrad import Tensor, nn
import numpy as np

class Linear_Model():
  def __init__(self):
    # This first layer is kinda big
    self.W1 = Tensor.randn(480*640*3, 600) * (1 / 480*640*3) ** 0.5
    self.b1 = Tensor.zeros(600)

    self.W2 = Tensor.randn(600, 400) * (1 / 600) ** 0.5
    self.b2 = Tensor.zeros(400)

    self.W3 = Tensor.randn(400, 200) * (1 / 400) ** 0.5
    self.b3 = Tensor.zeros(200)

    self.W4 = Tensor.randn(200, 100) * (1 / 200) ** 0.5
    self.b4 = Tensor.zeros(100)

    self.W5 = Tensor.randn(100, 1) * (1 / 100) ** 0.5
    self.b5 = Tensor.zeros(1)

  def __call__(self, X: Tensor) -> Tensor:
    X = X.flatten(1)
    X = X.matmul(self.W1).add(self.b1).tanh().dropout()
    X = X.matmul(self.W2).add(self.b2).tanh().dropout()
    X = X.matmul(self.W3).add(self.b3).tanh().dropout()
    X = X.matmul(self.W4).add(self.b4).tanh().dropout()
    X = X.matmul(self.W5).add(self.b5)
    return X

# Instantiate the model
model = Linear_Model()

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

from tinygrad import Tensor, nn
import numpy as np
import pdb
import psutil

cache_path="./data/flow_images.npy"

"""
X = np.load(cache_path)
  ...,
  [ 94, 255,   0],
  [ 94, 255,   0],
  [ 94, 255,   0]]]], shape=(20399, 480, 640, 3), dtype=uint8)
"""

#pdb.set_trace()

"""
Y = np.array([float(line) for line in open("./data/train.txt")])[: -1]
array([28.105569, 28.105569, 28.106527, ...,  2.289795,  2.292917,
2.2606  ], shape=(20399,))
"""

print(f"Defining network")
class Model():
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


model = Model()

print(f"Declaring optimizer")
optim = nn.optim.SGD(nn.state.get_parameters(model), lr=0.01)
#optim = nn.optim.Adam(params, lr=0.001)
"""SGD may be leading to overfit"""

# Load size
# Consider reducing to 500 and 16
#chunk = 1000
#batch = 32
chunk = 500
#batch = 16
batch = 32

n_chunk = 0
total_chunk = 20399 // chunk

print(f"Training")

total_loss = 0

print(f"Memory usage pre training: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
Y = np.array([float(line) for line in open("./data/train.txt")])[:-1]  # 20399 labels
#for chunk_X, chunk_Y in load_data_in_chunks():
# Try using 
for i in range (0, 20399, chunk):
  chunk_X = np.load("./data/flow_images.npy", mmap_mode='r')[i:i+chunk].astype(np.float32) / 255.0
  chunk_X = Tensor(chunk_X, dtype='float32')
  chunk_Y = Y[i:i + chunk]
  chunk_Y = Tensor(chunk_Y, dtype='float32')
  print(f"Memory usage outer loop: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
  Tensor.training=True
  
  # Mini-batch training within chunk
  chunk_loss = 0
  indices = np.random.permutation(chunk_X.shape[0])
  for j in range(0, chunk_X.shape[0], batch):
    print(f"Memory usage inner loop: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
    batch_idx = indices[j:j + batch].tolist()
    X_batch = chunk_X[batch_idx]
    Y_batch = chunk_Y[batch_idx]

    print("Doing forward")
    # Add l2 loss regularization
    # l2_loss = sum(p.square().sum() for p in params) * 0.01
    loss = model(X_batch).sub(Y_batch).square().mean() # + l2_loss # MSE Loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    chunk_loss += loss.numpy()
    print(f"Chunk {n_chunk + 1}-{total_chunk} % proccessed: {((j / chunk_X.shape[0]) * 100):.4f} | Loss: {loss.numpy():.4f}")

  chunk_loss /= (chunk // batch)
  print("="*40)
  print(f"Mean Chunk loss {chunk_loss:.4f}")
  n_chunk += 1
  print(f"Total data proccessed %: {(n_chunk / total_chunk):.4f}")
  print("="*40)

  total_loss += chunk_loss
  #gc.collect() # Check for garbaje collection

total_loss /= total_chunk
print(f"Mean total loss {total_loss}")

# Save model parameters
model_save_path = "./models/linear_sgd.npz"
print(f"Saving model to {model_save_path}")

# Get the model's parameters
param_dict = {}
for i, param in enumerate(nn.state.get_parameters(model)):
    param_dict[f'param_{i}'] = param.numpy()

# Save all parameters to a single .npz file
np.savez(model_save_path, **param_dict)
print(f"Model saved successfully to {model_save_path}")


"""
# Instantiate the model
model = Model()

# Load the saved model
model_save_path = "./model_weights.npz"
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
"""

"""
With RELU

Doing forward
Loss: 13507033.0
Doing forward
Loss: 3.371206309336986e+30
Doing forward
Loss: inf
Doing forward
Loss: nan
Doing forward

==============
Changed to TANH

Loss: 713.3308715820312
Doing forward
Loss: 122493.65625
Doing forward
Loss: 23008048.0
Doing forward
Loss: 5476477952.0
Doing forward
Loss: 1723970486272.0
Doing forward
Loss: 625388577357824.0
Doing forward
Loss: 2.0519178768849306e+17

==============
Normalize weigths

Loading data in chunks
Doing forward
Loss: 410.64361572265625
Doing forward
Loss: 93295.6953125
Doing forward
Loss: 18867530.0
Doing forward
Loss: 5181755904.0

==============
Changed neurons to 200

Doing forward
Loss: 450.91021728515625
Doing forward
Loss: 449739.84375
Doing forward
Loss: 418335872.0
Doing forward
Loss: 429689733120.0
Doing forward

================
Changed neurons to 300

Doing forward
Loss: 363.5306091308594
Doing forward
Loss: 763290.5
Doing forward
Loss: 1609939200.0
Doing forward
Loss: 3368920023040.0
Doing forward
Loss: 8773269029126144.0
Doing forward
Loss: 2.481696040229955e+19

Just incrementing the neurons does not seems to help learning

==================
With three layers deep, the learning improves but it stagnates at a local maximum

Loss: 432.3953857421875
Doing forward
Loss: 147.9228057861328
Doing forward
Loss: 191.9700927734375
Doing forward
Loss: 160.77333068847656
Doing forward
Loss: 169.14134216308594
Doing forward
Loss: 173.40576171875
Doing forward
Loss: 144.19711303710938
Doing forward
Loss: 168.5128631591797
Doing forward
Loss: 162.5296630859375
Doing forward
Loss: 162.90748596191406
Doing forward
Loss: 178.20960998535156
Doing forward
Loss: 153.54296875
Doing forward
Loss: 149.7224884033203

==================
4 Layer deep network

Declaring optimizer
Training
Loading data in chunks
Doing forward
Loss: 446.0902099609375
Doing forward
Loss: 13.301798820495605
Doing forward
Loss: 10307.1513671875
Doing forward
Loss: 223622.25
Doing forward
Loss: 57503032.0
Doing forward
Loss: 15093360640.0

==================
Reduce learning rate to 0.01 and got better results

Loss: 457.8882751464844
Doing forward
Loss: 234.8868408203125
Doing forward
Loss: 89.94364166259766
Doing forward
Loss: 76.10680389404297
Doing forward
Loss: 58.915340423583984
Doing forward
Loss: 52.985084533691406
Doing forward
Loss: 51.70574951171875
Doing forward
Loss: 49.62165451049805
Doing forward
Loss: 36.65394973754883
Doing forward
Loss: 26.314184188842773
Doing forward
Loss: 27.996845245361328
Doing forward
Loss: 36.57184982299805
Doing forward
Loss: 29.273866653442383
Doing forward
Loss: 24.36199188232422
Doing forward
Loss: 21.98240852355957
Doing forward
Loss: 24.80967903137207
Doing forward
Loss: 17.632614135742188
Doing forward
Loss: 13.12913703918457
Doing forward
Loss: 12.841031074523926
Doing forward
Loss: 7.322067737579346
Doing forward
Loss: 16.83162498474121
Doing forward
Loss: 16.8055419921875
Doing forward
Loss: 16.514108657836914
Doing forward
Loss: 12.736820220947266
Doing forward
Loss: 20.2390079498291
Doing forward
Loss: 25.58208465576172
Doing forward
Loss: 38.64585494995117
Doing forward
Loss: 30.62093734741211
Doing forward

Now the Goal is to see how low the loss can get using only a linear network
loss <= 10 would be nice but don't know if it is possible

=========================
With another layer performance increases but there is still high variability

Doing forward
I: 0 of 1000 | Loss: 436.9182434082031
Doing forward
I: 32 of 1000 | Loss: 195.4608917236328
Doing forward
I: 64 of 1000 | Loss: 62.779335021972656
Doing forward
I: 96 of 1000 | Loss: 77.89615631103516
Doing forward
I: 128 of 1000 | Loss: 75.02098846435547
Doing forward
I: 160 of 1000 | Loss: 70.41670989990234
Doing forward
I: 192 of 1000 | Loss: 63.474754333496094
Doing forward
I: 224 of 1000 | Loss: 33.384090423583984
Doing forward
I: 256 of 1000 | Loss: 27.371028900146484
Doing forward
I: 288 of 1000 | Loss: 45.24267578125
Doing forward
I: 320 of 1000 | Loss: 55.57011795043945
Doing forward
I: 352 of 1000 | Loss: 60.7623291015625
Doing forward
I: 384 of 1000 | Loss: 55.89450454711914
Doing forward
I: 416 of 1000 | Loss: 58.7736930847168
Doing forward
I: 448 of 1000 | Loss: 36.36652755737305
Doing forward
I: 480 of 1000 | Loss: 18.890789031982422
Doing forward
I: 512 of 1000 | Loss: 16.9257869720459
Doing forward
I: 544 of 1000 | Loss: 11.89849853515625
Doing forward
I: 576 of 1000 | Loss: 6.863011360168457
Doing forward
I: 608 of 1000 | Loss: 10.803991317749023
Doing forward
I: 640 of 1000 | Loss: 14.29101276397705
Doing forward
I: 672 of 1000 | Loss: 35.66658401489258
Doing forward
I: 704 of 1000 | Loss: 28.417312622070312
Doing forward
I: 736 of 1000 | Loss: 20.645124435424805
Doing forward
I: 768 of 1000 | Loss: 15.336779594421387
Doing forward
I: 800 of 1000 | Loss: 14.886085510253906
Doing forward
I: 832 of 1000 | Loss: 17.408824920654297
Doing forward
I: 864 of 1000 | Loss: 16.256032943725586
Doing forward
I: 896 of 1000 | Loss: 13.762035369873047
Doing forward
I: 928 of 1000 | Loss: 12.161652565002441
Doing forward
I: 960 of 1000 | Loss: 9.883871078491211
Doing forward
I: 992 of 1000 | Loss: 15.759051322937012

=========================
Reducing batch and chunk size by half reduced the CPU Ussage but the RAM usage kept the same.
The only thing is that reducing chunck size could reduce the context neeeded. We'll see.

So, with 2 epochs it is kinda overfitting but afterwars the loss starts going up again

It got to a very low loss

Memory usage inner loop: 13601.18 MB
Doing forward
%: 57.6000 | Loss: 0.0502
Memory usage inner loop: 13601.18 MB
Doing forward
%: 60.8000 | Loss: 0.0970
Memory usage inner loop: 13601.18 MB
Doing forward
%: 64.0000 | Loss: 0.0787
Memory usage inner loop: 13601.18 MB
Doing forward
%: 67.2000 | Loss: 0.0758
Memory usage inner loop: 13601.18 MB

=========================
Now we need to make it stop overfitting

Added some .5 drop out rate and maybe change optimizer to ADAM
and maybe add l2 regularization

if that does not works reduce the first layer params5

Just adding the dropout prevent it from overfitting.
This could be used to generate a model based only on linear layers

"""
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
#
#file_npz = "./inouts3/npz/000_0000.npz" # on my windows desktop
file_npz = "/panfs/ccds02/nobackup/people/skorkin/npz/000_0000.npz" # on my Adapt
#
# Adjustable parameters:
num_neurons = 512
batch_size = 51891840 // 4 # [51891840_cases * 8_input_params * 1_output_param] 32bit floats
#
num_epochs = 3
learning_rate = 0.01
#
# Fixed parameters:
num_inp = 8
num_out = 1
print(f"Memory wanted: {(num_inp + num_out) * batch_size * 4.0/(1024 ** 3):.3f} GB")

#
# Execution starts here:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

#==============================VRAM============================================
# Get the total memory of the GPU
device_index = 0
total_memory = torch.cuda.get_device_properties(device_index).total_memory
print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")

# Check initial GPU memory usage
allocated_memory_initial = torch.cuda.memory_allocated(device_index)
reserved_memory_initial = torch.cuda.memory_reserved(device_index)
print(f"Initially allocated GPU memory: {allocated_memory_initial / (1024 ** 3):.2f} GB")
print(f"Initially reserved GPU memory: {reserved_memory_initial / (1024 ** 3):.2f} GB")

# Set the memory fraction
vram_fraction = 0.99
print(f"Requested GPU memory: {total_memory * vram_fraction / (1024 ** 3):.2f} GB")
torch.cuda.set_per_process_memory_fraction(vram_fraction, device_index)  # Use 99% of the total GPU memory

# Clear the cache before starting the training
torch.cuda.empty_cache()

# Check memory usage after clearing the cache
allocated_memory_after_cache_clear = torch.cuda.memory_allocated(device_index)
reserved_memory_after_cache_clear = torch.cuda.memory_reserved(device_index)
print(f"Allocated GPU memory after clearing cache: {allocated_memory_after_cache_clear / (1024 ** 3):.2f} GB")
print(f"Reserved GPU memory after clearing cache: {reserved_memory_after_cache_clear / (1024 ** 3):.2f} GB")
#==============================END of VRAM=====================================

model = nn.Sequential( nn.Linear(num_inp, num_neurons),     nn.Sigmoid(),
                       nn.Linear(num_neurons, num_neurons), nn.Sigmoid(),
                       nn.Linear(num_neurons, num_neurons), nn.Sigmoid(),
                       nn.Linear(num_neurons, num_out) ).to(device)
print("model is on:", next(model.parameters()).device)
#
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
#
npz_data = np.load(file_npz)
xtrain = npz_data['data_inp'] # xtrain.shape = (51891840, 8)
ytrain = npz_data['data_out'] # xtrain.shape = (51891840, 1)
#
shuffle = True
if shuffle:
    nrows_x = xtrain.shape[0]
    indices = np.random.permutation(nrows_x)
    # rand_indx = torch.randperm(len(xtrain))
    xtrain = xtrain[indices, :]
    ytrain = ytrain[indices, :]
#
xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32).to(device)
ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32).to(device)
print("xtrain_tensor is on:", xtrain_tensor.device)
print("ytrain_tensor is on:", ytrain_tensor.device)
#
t0 = time.time()
for epoch in range(num_epochs):
    for i in range(0, xtrain_tensor.size(0), batch_size):
        inputs = xtrain_tensor[i:i+batch_size, :]
        targets = ytrain_tensor[i:i+batch_size, 0]
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1)) # -1 - use 1st dimension as in original aray; 1 - second dimension must be 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: %d" %epoch)
dt_sec = time.time() - t0
print("dt_sec (raw):", dt_sec)
time_hhmmss = str(timedelta(seconds=dt_sec))
print("training time:", time_hhmmss)
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
#
#file_npz = "./inouts3/npz/000_0000.npz" # on my windows desktop
file_npz = "/panfs/ccds02/nobackup/people/skorkin/npz/000_0000.npz" # on my Adapt
#
# Adjustable paramters:
num_epochs = 2
batch_size = 1024 # total = 51891840, num_batches = total/batch_size
#
# Fixed paramters:
num_inp = 8
num_neurons = 5
num_out = 1
learning_rate = 0.01
#
# Execution starts here:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("what is my device? it is ...", device)
model = nn.Sequential( nn.Linear(num_inp, num_neurons),     nn.Sigmoid(),
                       nn.Linear(num_neurons, num_neurons), nn.Sigmoid(),
                       nn.Linear(num_neurons, num_neurons), nn.Sigmoid(),
                       nn.Linear(num_neurons, num_out) ).to(device)
print("where is my model? on ...", next(model.parameters()).device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
#
npz_data = np.load(file_npz)
xtrain = npz_data['data_inp'] # xtrain.shape = (51891840, 8)
ytrain = npz_data['data_out'] # xtrain.shape = (51891840, 1)
#
xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32).to(device)
ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32).to(device)
print("where are my tensors?")
print("-xtrain_tensor is on ...", xtrain_tensor.device)
print("-ytrain_tensor is on ...", ytrain_tensor.device)
dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
t0 = time.time()
inputs_count = 0
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1)) # -1 - use 1st dimension as in original aray; 1 - second dimension must be 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        inputs_count += 1
        if inputs_count % 1000 == 0:
            print("epoch & inputs_counter %d %d" %(epoch, inputs_count))
dt_sec = time.time() - t0
print("done! elapsed time (minutes): %.1f" %(dt_sec/60.))
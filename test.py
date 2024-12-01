import torch.nn as nn


model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Sigmoid()
)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, label)

model.
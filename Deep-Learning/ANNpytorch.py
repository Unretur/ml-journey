import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. DATA
# ==========================================

X = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0]
], dtype=torch.float32)


y = torch.tensor([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1]
], dtype=torch.float32)


# ==========================================
# 2. ANN ARCHITECTURE
# ==========================================

class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        # Input → Hidden layer
        self.layer1 = nn.Linear(2, 8)

        # Hidden → Hidden
        self.layer2 = nn.Linear(8, 4)

        # Hidden → Output
        self.output = nn.Linear(4, 1)


    def forward(self, x):

        # First layer
        x = self.layer1(x)

        # Activation
        x = torch.relu(x)

        # Second layer
        x = self.layer2(x)

        # Activation
        x = torch.relu(x)

        # Output layer
        x = self.output(x)

        return x


# ==========================================
# 3. CREATE MODEL
# ==========================================

model = ANN()

print(model)


# ==========================================
# 4. LOSS FUNCTION
# ==========================================

loss_function = nn.BCEWithLogitsLoss()


# ==========================================
# 5. OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. TRAINING
# ==========================================

epochs = 1000


for epoch in range(epochs):

    # Forward propagation
    output = model(X)

    # Calculate loss
    loss = loss_function(output, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()


    # Print loss
    if epoch % 100 == 0:

        print(
            f"Epoch: {epoch}, "
            f"Loss: {loss.item():.4f}"
        )


# ==========================================
# 7. PREDICTION
# ==========================================

model.eval()


with torch.no_grad():

    # Get raw outputs
    logits = model(X)

    # Convert logits to probability
    probabilities = torch.sigmoid(logits)

    # Convert probability to class
    predictions = (
        probabilities >= 0.5
    ).float()


# ==========================================
# 8. RESULTS
# ==========================================

print("\nProbabilities:")
print(probabilities)


print("\nPredictions:")
print(predictions)


print("\nActual:")
print(y)
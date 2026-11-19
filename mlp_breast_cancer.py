import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #utilisation GPU nvidia (cuda) si disponible, sinon du CPU


data = load_breast_cancer()
X, y = data.data, data.target  #on considère 0 comme classe positive et 1 comme classe négative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #application de la normalisation sur les données d'entraînement
X_test = scaler.transform(X_test) #application de la normalisation sur les données de test, en utilisant les paramètres calculés sur les données d'entraînement


train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), #float 32 bits = standard pour les tenseurs de PyTorch
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class MLP(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),)
    
  def forward(self, x: torch.Tensor):
    return self.net(x)


model = MLP(input_dim=X_train.shape[1]).to(device) #X_train.shape[1] = nombre de features du dataset breast cancer, ici 30. .to(device) = envoie le modèle sur GPU si disponible, sinon CPU
criterion = nn.BCEWithLogitsLoss() #calcul de la fonction perte cross entropy binaire, sigmoide déjà appliquée dans la fonction de perte donc pas besoin de l'appliquer dans le forward du modèle
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) #algo adam pour optimisation des poids, weight_decay = régularisation L2 pour éviter overfitting, utilise le calcul des gradients pour mettre à jour les poids du modèle


epochs = 50
for epoch in range(1, epochs + 1):
  model.train()
  train_loss = 0.0

  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad() #remise à 0 des gradients pour éviter accumulation de gradients d'une itération à l'autre
    logits = model(batch_X)
    loss = criterion(logits, batch_y)
    loss.backward() #calcul des gradients
    optimizer.step()

    train_loss += loss.item() * batch_X.size(0)

  train_loss /= len(train_dataset)

  if epoch % 10 == 0 or epoch == 1:
    print(f"epoque {epoch}/{epochs}, perte (train) : {train_loss}")

model.eval()
all_preds = []
all_targets = []

with torch.no_grad(): #torch.no_grad() = désactive le calcul des gradients pour économiser de la mémoire et accélérer l'inférence
  for batch_X, batch_y in test_loader:
    batch_X = batch_X.to(device)
    logits = model(batch_X)
    probs = torch.sigmoid(logits) #sert à transformer les logits en probabilités (0>x>1) car la sortie du modèle est un logit (valeur non bornée) et on veut une probabilité pour classification binaire
    preds = (probs >= 0.5).int().cpu().numpy()

    all_preds.extend(preds.flatten())
    all_targets.extend(batch_y.numpy().flatten().astype(int))

acc = accuracy_score(all_targets, all_preds)
print(f"\naccuracy sur le jeu de test : {acc} ({acc * 100} %)")

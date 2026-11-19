PRISE EN MAIN PYTORCH : MLP (jeu breast-cancer sklearn)

Ce dépôt contient mon premier réseau de neurones sous pytorch. Je l'ai écrit en découvrant pytorch et en étudiant la théorie en parallèle.

Objectif : Entraîner un perceptron multicouche pour une classification binaire sur un petit jeu tabulaire, afin de me familiariser avec la pytorch (tenseurs, nn.Module, fonction de perte, optimiseur, boucle d'entraînement, évaluation)

Données : jeu breast cancer de sklearn (sklearn.datasets.load_breast_cancer) de taille (569,30), classification binaire (tumeur maligne/bénigne)

Séparation entrainement/test : test = 20%, entrainement 80%, graine aléatoire = 50
Prétraitement: normalisation avec transform() sur data test et fit.transform() sur train test

Modèle et entraînement : Architecture : mlp à 2 hidden layers, entrée 30 -> 64 -> 32 -> sortie 1 pour classification binaire (avec dropout(0.2) après la 1re activation pour éviter surapprentissage)
Il s'entraîne pendant 50 époques par petits lots (batchs de 32 à l'entraînement, 64 au test) avec l'optimiseur Adam (learning rate=0.001 et une légère régularisation weight_decay=1e-4). La fonction de perte BCEWithLogitsLoss mesure l'erreur en intégrant directement la sigmoïde pour plus de stabilité numérique.

Resultats: accuracy sur le jeu de test : 0.9824561403508771 (98.24561403508771 %)

Limites: petit jeu de données donc résultats ne disent rien de la performance sur d'autres problèmes.
Le gradient est calculé par autograd (loss.backward(), que j'utilise ici comme une boîte noire, j'étudie la rétropropagation et la descente de gradient en ce moment.

Une partie du code s'appuie sur la documentation officielle PyTorch (couches, perte, optimiseur) ainsi que les cours en ligne de sentdex (youtube).

Étude en cours : Goodfellow, Deep Learning chapitres 4, 6 et 8 ; Stanford CS231n 

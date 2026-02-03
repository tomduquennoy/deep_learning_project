# Deep Learning Project - Graph Neural Networks (GNN)

Ce repository est un projet de deep learning. Le but est d’introduire le concept de GNN (Graph Neural Networks). 

Il y a à disposition : 
- une vidéo d’introduction aux GNN : https://youtu.be/ks3VZtoXITw
- la présentation .pdf
- un code python correspondant à l'application d'un GCN à un ensemble de graphes moléculaires ZINC-250K.

Inspiré de l'article suivant : https://distill.pub/2021/gnn-intro/

## Implémentation python

Pour l'implémentation, différents éléments sont disponibles.

Les codes à exécuter :
- main_molecules_graph_regression.py est le code main : son exécution entraîne le modèle GCN
- plot_training_curves.py permet de plot les courbes d'entraînement (avec loss)
- visualize_molecules.py permet de visualiser un échantillon de molécules de train et test 

Les outils / résultats : 
- data contient les données. Normalement, en exécutant le main, on exécute un fichier qui télécharge la base de donnée source (1.3Go).
- les autres dossiers contiennents des outils ou des résultats (d'entraînement, de test)
- 4 plots en .png sont disponibles directement pour observer les résultats de l'entraînement du modèle
- un fichier result.txt contient les informations de fin d'epochage.

Je ne vous conseille pas de lancer le code. Le code à lancer est assez long (30min sur mon pc). 
Sinon, il faut exécuter le fichier python main_molecules_graph_regression.py, puis les autres codes (plot_training_curves.py, visualize_molecules.py) pour avoir la data visualisation.

Les packages suivants sont à installer en amont :

` pip install torch dgl numpy matplotlib tensorboardX tensorboard tqdm networkx `

## Fonctionnement du modèle

Le Dataset est ensemble de graphes moléculaires où les noeuds sont des atomes (C, N, O...) encodé par [numéro atomique, charge, aromaticité] ; les arêtes sont des liaisons chimiques (simple, double, triple, aromatique). La cible est une valeur scalaire continue mesurant l’hydrophobie d’une molécule.

La Pipeline se développe en 3 étapes : 
- Embedding Layer : projection des atomes dans un espace continu
- GCN (ou GIN) : entre 4 et 8 couches pour ZINC. Chaque atome "apprend" son environnement chimique. Un carbone au sein d'un cycle benzénique doit avoir un embedding différent d'un carbone dans une chaîne linéaire.
- Readout (Global Pooling) : fusion de tous les embeddings finaux en un seul vecteur représentant la molécule entière.

Enfin on passe ce seul vecteur dans un MLP (Multi-Layer Perceptron) pour avoir une valeur unique.

Les résultats sont les suivants : 
- Test MAE: 0.4057
- Train MAE: 0.3026
- Learning rate final : 1.56e-5

Le modèle permet de prendre en compte une grande complexité de liaisons.
L’agrégation de voisinage permet au réseau de 'peser' l'importance de chaque type d'atome et de sa position dans la structure pour prédire la solubilité.
Le GNN parvient à différencier des molécules ayant un nombre d'atomes similaire mais des connectivités différentes (ex: Molécules #252 et #521, toutes deux à 23 atomes).

# deep_learning_project

Ce repository est un projet de deep learning. Le but est d’introduire le concept de GNN (Graph Neural Networks). 

Il y a à disposition : 
- une vidéo d’introduction aux GNN : https://youtu.be/ks3VZtoXITw
- la présentation .pdf
- un code python correspondant à l'application d'un GCN à un ensemble de graphes moléculaires ZINC-250K.

Le code à lancer est assez long (30min sur mon pc), il faut exécuter le fichier python main_molecules_graph_regression.py, puis les autres codes (plot_training_curves.py, visualize_molecules.py) pour avoir la data visualisation.

Les packages suivants sont à installer en amont :

$ pip install torch dgl numpy matplotlib tensorboardX tensorboard tqdm networkx 

Inspiré de l'article suivant : https://distill.pub/2021/gnn-intro/

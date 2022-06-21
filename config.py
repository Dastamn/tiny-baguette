import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 50
BATCH_SIZE = 64
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 1024

LR = .001
DROPOUT = .5
NUM_LAYERS = 4

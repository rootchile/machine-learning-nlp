import torch
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self, embedding):
        """
        Args:
            embedding ([narray]): Vector of all words
        """
        super(LSTM, self).__init__()
        
        num_words = embedding.shape[0]
        embed_dim = embedding.shape[1]
        
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )    
        
        self.embedding.weights = nn.Parameter(torch.tensor(embedding, dtype=torch.float32))
        # we dont train pretrained embeddings
        self.embedding.weights.requires_grad = False
        
        #LSTM
        self.lstm = nn.LSTM(embed_dim, 128, birectional=True, batch_first=True)
        self.out  = nn.Linear(512,1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        
        out = torch.cat((avg_pool, max_pool),1)
        out = self.out(out)
        
        return out
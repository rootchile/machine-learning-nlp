import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Args:
            reviews ([narray]): array of reviews
            targets ([narray]): vector of targets
        """
        self.reviews = reviews
        self.targets = targets
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        rewiew = self.rewiews[item, :]
        target = self.targets[item]
        
        return {'review': torch.tensor(rewiew, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.float)
               }    
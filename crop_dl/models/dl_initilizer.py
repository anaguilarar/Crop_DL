
import torch
from .dl_architectures import get_instance_segmentation_model
import os

class DLInstanceModel:
    
    def __init__(self, weights = None, modeltype = "instance_segmentation",
                 lr = 0.005, device = None) -> None:
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if modeltype == "instance_segmentation":
            model = get_instance_segmentation_model(2).to(self.device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=lr,
                                        momentum=0.9, weight_decay=0.0005)

        if weights:
            if os.path.exists(weights):
                if self.device == "cpu":              
                    model_state, optimizer_state = torch.load(
                    weights, map_location=self.device)
                else:
                    model_state, optimizer_state = torch.load(
                    weights)
                
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("weights loaded")
            
        self.model = model
        self.optimizer = optimizer



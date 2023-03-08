
import torch
from crop_dl.dl_architectures import get_instance_segmentation_model
import os

class DLInstanceModel:
    
    def __init__(self, weights = None, modeltype = "instance_segmentation") -> None:
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if modeltype == "instance_segmentation":
            maskrcnnmodel = get_instance_segmentation_model(2).to(self.device)
            
        params = [p for p in maskrcnnmodel.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        if weights:
            if os.path.exists(os.path.join(weights,"checkpoint")):
                model_state, optimizer_state = torch.load(
                    os.path.join(weights, "checkpoint"))
                
            maskrcnnmodel.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        self.maskrcnnmodel = maskrcnnmodel
        self.optimizer = optimizer



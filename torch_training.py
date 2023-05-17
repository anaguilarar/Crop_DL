import numpy as np
import tqdm
#import torch_tb_profiler
import torch

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

@torch.no_grad()
def eval_fn(model, dataloader, 
             loss_fn,
             epoch,
             progressbar = True):
    
    if progressbar:
        loop = tqdm.tqdm(dataloader, desc = f'Epoch {epoch}', leave=False)
    else:
        loop = dataloader      
    
    loop = tqdm.tqdm(dataloader, desc = f'Epoch {epoch}', leave=False)
    g_loss = 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    for idx, (images, targets) in enumerate(loop):
        images = list(img.to(device) for img in images)
        output = model(images)
            #loss = loss_fn(output, y)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        loss = mae(y.to('cpu').detach().numpy(), output.cpu().detach().numpy())   
        
        g_loss += loss.item()

    
    return g_loss/dataloader.__len__()

def train_msksegmen(model, 
                 dataloader, 
             optimizer,
             epoch,
             progressbar = True):
    
    if progressbar:
        loop = tqdm.tqdm(dataloader, desc = f'Epoch {epoch}', leave=False)
    else:
        loop = dataloader    
    g_loss = 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model.train()
    for idx, (imgtr, targets) in enumerate(loop):
        
        images = list(image.to(device) for image in imgtr)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images, targets)
            loss = sum(loss for loss in output.values())
            
            #loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()
        
        g_loss += loss.item()

    
    return g_loss/dataloader.__len__()
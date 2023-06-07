class SegmentationPrediction():
    
    @check_output_fn
    def load_weights(self, path, fn, suffix = 'pth.tar'):
        
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer"])
        print("weights loaded")
    
    def _original_size(self):
        msksc = [0]* len(list(self.msks))
        
        for i in range(len(self.msks)):
            msksc[i] = cv2.resize(self.msks[i][0], 
                                  [self.img.shape[2],self.img.shape[1]], 
                                  interpolation = cv2.INTER_AREA)  
       
        self.msks = np.array(np.expand_dims(msksc, axis =0))
        
    def get_mask(self, image, keepdims = True):
        
        self.imgtensor = image_to_tensor(image=image, outputsize = self.inputimgsize)
        self.img = image
        
        self.model.eval()
        with torch.no_grad():
            msks = self.model(torch.stack([self.imgtensor.to(self.device)]))
            
        if type(msks) is collections.OrderedDict:
            #self.msks = self.msks['out']
            self.msks =msks['out'].mul(255).byte().cpu().numpy()
        else:
            self.msks = msks.mul(255).byte().cpu().numpy()
        
        if keepdims:
            self._original_size()
        
        return self.msks

    def set_model(self):
        
        if self.arch == "Unet256":
            model = Unet256(in_channels=3, out_channels=1)
        
        return model.to(self.device)
    
    def __init__(self, architecture = "Unet256", configuration = {'lr':2e-4,
                                                                  'beta': (0.5, 0.999)}, device = None) -> None:
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.arch = architecture
        self.model = self.set_model()
        self.inputimgsize = (256, 256)
        self.opt = optim.Adam(self.model.parameters(), 
                      lr=configuration['lr'], betas=configuration['beta'])

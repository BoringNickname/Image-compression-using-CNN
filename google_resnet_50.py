#%%
from torchvision import models

deeplab = models.segmentation.deeplabv3_resnet50(pretrained=0, progress=1, num_classes=2)

class ImgSegModel(nn.Module):
    def __init__(self):
        super(ImgSegModel, self).__init__()
        self.dl = deeplab

    def forward(self,x):
        y = self.dl(x)['out']
        return y

class SegDataset(Dataset):
    
    def __init__(self, parentDir, imageDir, maskDir):
        self.imageList = glob.glob(parentDir+'/'+imageDir+'/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir+'/'+maskDir+'/*')
        self.maskList.sort()

    def __getitem__(self, index):
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        X = Image.open(self.imageList[index]).convert('RGB')
        X = preprocess(X)
        
        trftensor = transforms.ToTensor()
        
        yimg = Image.open(self.maskList[index]).convert('L')
        y1 = trftensor(yimg)
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        
        return X, y
            
    def __len__(self):
        return len(self.imageList)
# %%

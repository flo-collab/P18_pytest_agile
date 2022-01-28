import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pickle 

dev = torch.device("cpu")
cpu = torch.device("cpu")
np.random.seed(551)

# Stuff : 
list_stuff_path = 'list_stuff.sav' 
list_stuff = pickle.load(open(list_stuff_path, 'rb'))
dataDir,classNames,numClass,imageFiles,nnumEach,imageFilesList,imageClass,numTotal,imageWidth, imageHeight = list_stuff[0],list_stuff[1],list_stuff[2],list_stuff[3],list_stuff[4],list_stuff[5],list_stuff[6],list_stuff[7],list_stuff[8],list_stuff[9]

class MedNet(nn.Module):
    def __init__(self,xDim,yDim,numC): # Pass image dimensions and number of labels when initializing a model   
        super(MedNet,self).__init__()  # Extends the basic nn.Module to the MedNet class
        # The parameters here define the architecture of the convolutional portion of the CNN. Each image pixel
        # has numConvs convolutions applied to it, and convSize is the number of surrounding pixels included
        # in each convolution. Lastly, the numNodesToFC formula calculates the final, remaining nodes at the last
        # level of convolutions so that this can be "flattened" and fed into the fully connected layers subsequently.
        # Each convolution makes the image a little smaller (convolutions do not, by default, "hang over" the edges
        # of the image), and this makes the effective image dimension decreases.
        
        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

        # nn.Conv2d(channels in, channels out, convolution height/width)
        # 1 channel -- grayscale -- feeds into the first convolution. The same number output from one layer must be
        # fed into the next. These variables actually store the weights between layers for the model.
        
        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

        # These parameters define the number of output nodes of each fully connected layer.
        # Each layer must output the same number of nodes as the next layer begins with.
        # The final layer must have output nodes equal to the number of labels used.
        
        fcSize1 = 400
        fcSize2 = 80
        
        # nn.Linear(nodes in, nodes out)
        # Stores the weights between the fully connected layers
        
        self.ful1 = nn.Linear(numNodesToFC,fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2,numC)
        
    def forward(self,x):
        # This defines the steps used in the computation of output from input.
        # It makes uses of the weights defined in the __init__ method.
        # Each assignment of x here is the result of feeding the input up through one layer.
        # Here we use the activation function elu, which is a smoother version of the popular relu function.
        
        x = F.elu(self.cnv1(x)) # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x)) # Feed through second convolutional layer, apply activation
        x = x.view(-1,self.num_flat_features(x)) # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x)) # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x)) # Feed through second FC layer, apply output
        x = self.ful3(x)        # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

toTensor = tv.transforms.ToTensor()
def scaleImage(x):          # Pass a PIL image, return a tensor
    y = toTensor(x)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z

def preprocess_image(path):
    my_imagetensor = scaleImage(Image.open(path))
    return my_imagetensor

def get_accuracy_confmat_ontest():
    folder_tensor = 'pickels_tensors/'
    filename1 = 'imageTensor.sav'
    filename2 = 'classTensor.sav'
    imageTensor = pickle.load(open(folder_tensor+filename1, 'rb'))
    classTensor = pickle.load(open(folder_tensor+filename2, 'rb'))

    model = MedNet(imageWidth,imageHeight,numClass).to(dev)
    m_state_dict = torch.load('model_mednist_statedict_9983.pt', map_location=torch.device('cpu'))
    model.load_state_dict(m_state_dict)

    validFrac = 0.1   # Define the fraction of images to move to validation dataset
    testFrac = 0.1    # Define the fraction of images to move to test dataset
    validList = []
    testList = []
    trainList = []

    for i in range(numTotal):
        rann = np.random.random() # Randomly reassign images
        if rann < validFrac:
            validList.append(i)
        elif rann < testFrac + validFrac:
            testList.append(i)
        else:
            trainList.append(i)
            
    nTrain = len(trainList)  # Count the number in each set
    nValid = len(validList)
    nTest = len(testList)
    print("Training images =",nTrain,"Validation =",nValid,"Testing =",nTest)
   #training, validation, and testing tensors
    testIds = torch.tensor(testList)
    testX = imageTensor[testIds,:,:,:]
    testY = classTensor[testIds]

    batchSize = 300           # Batch size. Going too large will cause an out-of-memory error.
    testBats = -(-nTest // batchSize)     # Testing batches. Round up to include all

    confuseMtx = np.zeros((numClass,numClass),dtype=int)    # Create empty confusion matrix
    model.eval()
    with torch.no_grad():
        permute = torch.randperm(nTest)                     # Shuffle test data
        testX = testX[permute,:,:,:]
        testY = testY[permute]
        for j in range(testBats):                           # Iterate over test batches
            batX = testX[j*batchSize:(j+1)*batchSize,:,:,:].to(dev)
            batY = testY[j*batchSize:(j+1)*batchSize].to(dev)
            yOut = model(batX)                              # Pass test batch through model
            pred = yOut.max(1,keepdim=True)[1]              # Generate predictions by finding the max Y values
            for j in torch.cat((batY.view_as(pred), pred),dim=1).tolist(): # Glue together Actual and Predicted to
                confuseMtx[j[0],j[1]] += 1                  # make (row, col) pairs, and increment confusion matrix
    correct = sum([confuseMtx[i,i] for i in range(numClass)])   # Sum over diagonal elements to count correct predictions
    print("Correct predictions: ",correct,"of",nTest)
    print("Confusion Matrix:")
    print(confuseMtx)
    print(classNames)
    accuracy = correct/nTest
    print('accuracy:',accuracy)

    return accuracy, confuseMtx
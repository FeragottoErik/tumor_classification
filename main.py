from tables import *  # needed to read .h5 files
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from custom_functions.custom_resnet50 import resnet50
import pickle

PLOT=True #bool to plot images

h5file = open_file('/home/erikfer/pw1_chiara/MalignancyClassification.h5', mode='r')


y = np.array(h5file.root.Malignancy)
print(y)
print(len(y))
IDX_class_zero = np.where(y == 0)[0] # 0 is the class of maglinancy
IDX_class_one = np.where(y == 1)[0] # 1 is the class of benignancy (non-malignant)

radiomics_feats_names = []
with (open("/home/erikfer/pw1_chiara/Radiomics_features_names.pickle", "rb")) as openfile:
    while True:
        try:
            radiomics_feats_names.append(pickle.load(openfile))
        except EOFError:
            break

for element in radiomics_feats_names[0]:
    print(element) #this is just the list of the names of the features, not the features themselves

def normalize(x_img):
    MIN_BOUND = -1200.0
    MAX_BOUND = 600.0
    PIXEL_MEAN = 0.12  # 0.25
    PIXEL_CORR = int((MAX_BOUND - MIN_BOUND) * PIXEL_MEAN)  # in this case, 350
    x_img -= PIXEL_CORR  # centering
    x_img[x_img > MAX_BOUND] = MAX_BOUND  # clipping
    x_img[x_img < MIN_BOUND] = MIN_BOUND  # scaling
    x_img = ((x_img - (MIN_BOUND)) / (MAX_BOUND - (MIN_BOUND)))

    return x_img

# Display examples of the dataset
if PLOT:
    plt.figure(figsize=(20, 8))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        TPidx = random.randrange(len(IDX_class_one))
        # extraction of the central slice of the volumetric patch 26x48x48.
        image = normalize(h5file.root.images[IDX_class_one[TPidx], 12, :, :])
        plt.imshow(image, cmap='gray')
        plt.xlabel('Non-Malignant', fontsize=15)
        plt.xticks([])
        plt.yticks([])
        print(IDX_class_one[TPidx])
        plt.subplot(2, 5, i+6)
        FPidx = random.randrange(len(IDX_class_zero))
        # extraction of the central slice of the volumetric patch 26x48x48.
        image = normalize(h5file.root.images[IDX_class_zero[FPidx], 12, :, :])
        plt.imshow(image, cmap='gray')
        plt.xlabel('Malignant', fontsize=15)
        plt.xticks([])
        plt.yticks([])
        print(IDX_class_zero[FPidx])

    image = normalize(h5file.root.images[IDX_class_zero[FPidx], 23, :, :])
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

#store patients ID-to-index in a text file
anonID_to_index = dict()
for i, n in enumerate(h5file.root.anonID):
    print(i, n)
    if n not in anonID_to_index.keys():
        anonID_to_index[n] = [i]
    else:
        anonID_to_index[n].append(i)
#save the list to a text file in '/home/erikfer/Detection_project/CODE' dir
with open('anonID_to_index.txt', 'w') as f:
    for key, value in anonID_to_index.items():
        f.write('%s:%s\n' % (key, value))
    
#randomly split patients into train, validation and test sets
random.seed(1234)
patients = list(anonID_to_index.keys())
random.shuffle(patients)
train_patients = patients[:int(0.7*len(patients))]
val_patients = patients[int(0.7*len(patients)):int(0.85*len(patients))]
test_patients = patients[int(0.85*len(patients)):]
print('train patients:', len(train_patients))
print('val patients:', len(val_patients))
print('test patients:', len(test_patients))

#Create a dataset object that given the list of patient ID, creates a dataset of images and labels
#where images are given by h5file.root.images[indx, :, :, :] with indx in anonID_to_index[ID] for each patient ID
#and labels are given by h5file.root.Malignancy[indx] with indx in anonID_to_index[ID] for each patient ID
class Dataset(object):
    def __init__(self, patients):
        self.patients = patients
        self.images = []
        self.labels = []
        self.anonID = []
        self.radiomix_feats = []
        for p in patients:
            for i in anonID_to_index[p]:
                image = h5file.root.images[i, 12, :, :] #TODO: instead of selecting slice 12, leave ':' to select the whole
                #volume, then use a 3D resnet or something liek that to process 3D-inputs, it will change the output shape
                #of the layer before the pooling, the feature map will probably be 4D and not 3D
                image = np.expand_dims(image, axis=0)  # Reshape image to (1, x, y)
                #TODO: normalize the image with image = normalize(image)
                self.images.append(image)
                self.labels.append(h5file.root.Malignancy[i])
                self.anonID.append(h5file.root.anonID[i])
                self.radiomix_feats.append(h5file.root.radiomics_features[i, :])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.anonID = np.array(self.anonID)
        self.radiomix_feats = np.array(self.radiomix_feats)
        print('Dataset created for patients:', len(patients))
        print('Dataset size:', len(self.images))
        print('Dataset labels:', self.labels.shape)
        print('Dataset anonID:', self.anonID.shape)
        print('Dataset radiomix_feats:', self.radiomix_feats.shape)
        count_0 = np.count_nonzero(self.labels == 0)
        count_1 = np.count_nonzero(self.labels == 1)
        print("Occurrences of 0:", count_0)
        print("Occurrences of 1:", count_1)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index): #doing dataset[index] will return the image and the label 
        #of the index-th patient and the rediomics frature of the image
        return self.images[index], self.labels[index], self.anonID[index], self.radiomix_feats[index]
    
#it is better to split the dataset according tu patient ID and not according to the index of the images since 
#same patient will have similar images and we want to avoid having the same patient in both train and validation set
train_dataset = Dataset(train_patients)
val_dataset = Dataset(val_patients)
test_dataset = Dataset(test_patients)

#basically resnet50 but it returns not only the output of the last layer but also the output of the pooling layer
#to enable to further concat the embeddings with the radiomix features
resnet_model= resnet50(pretrained=False, num_classes=2)

#defining a FCNN to binary classify input data of shape [batch_size, 190+2080] (which is the concat between
# each image output from resnet, just after the pooling layer before fc layers and softmax)
# in two classes 0 and 1 leaving logits as output
# TODO: move the model definition to another file to make it more readable and clean
class FCNN(torch.nn.Module):
    def __init__(self, input_feats=2238, num_classes=2): #input_feats is the number of features of the input that may vary if you 
        #chose another set of radiomics features or if the output of the classification model (let's say is no more resnet) changes
        #in shape or number of features
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_feats, input_feats//2)
        self.fc2 = torch.nn.Linear(input_feats//2, 50)
        self.fc3 = torch.nn.Linear(50, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=1) #used only ad classification time when predicting the class
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        outs = self.softmax(logits)
        return outs
    
FCNN_model = FCNN(num_classes=2)
#basic training loop for the model using train_dataset and val_dataset
def train_model(model, train_dataset, val_dataset, batch_size=32, num_epochs=10, learning_rate=0.001):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for i, (images_greyscale, labels, anonID, radiomix) in enumerate(train_loader):
            for i, (images_greyscale, labels, anonID, radiomix) in enumerate(train_loader):
                if round(i/len(train_loader)*100, 2) % 10 == 0: #printing info each 10% of the training in every epoch
                    print(round(i/len(train_loader)*100, 2), '%', end='\r')
                images_greyscale = images_greyscale.float()
                labels = labels.long() 
                images = np.repeat(images_greyscale[...], 3, 1) #the input needs to be RGB so we repeat the greyscale image 3 times
                emb, outputs = model(images) #at index 1 there is the output of the last layer while at 0 the embeddings after the pooling layer
                #emb is [batch_size, 2048, 1, 1], it is reshaped to [batch_size, 2048] and then concatenated with radiomix_feats so as to have a tensor of shape [batch_size, 2048+190]
                input_FCNN = torch.cat((emb.reshape(emb.shape[0], emb.shape[1]), radiomix.float()), dim=1)
                outputs = FCNN_model(input_FCNN)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                epoch_loss += loss.item()
                epoch_acc += torch.sum(preds == labels.data)
            images_greyscale = images_greyscale.float()
            labels = labels.long() 
            images = np.repeat(images_greyscale[...], 3, 1) #the input needs to be RGB
            emb, outputs = model(images) #at index 1 there is the output of the last layer while at 0 the embeddings after the pooling layer
            #emb is [batch_size, 2048, 1, 1], it is reshaped to [batch_size, 2048] and then concatenated with radiomix_feats so as to have a tensor of shape [batch_size, 2048+190]
            input_FCNN = torch.cat((emb.reshape(emb.shape[0], emb.shape[1]), radiomix.float()), dim=1)
            outputs = FCNN_model(input_FCNN)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item()
            epoch_acc += torch.sum(preds == labels.data)
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = epoch_acc.double() / len(train_loader.dataset)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print('Epoch:', epoch, 'Train loss:', epoch_loss, 'Train acc:', epoch_acc)
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        for i, (images_greyscale, labels, anonID, radiomix) in enumerate(val_loader):
            images_greyscale = images_greyscale.float()
            images = np.repeat(images_greyscale[...], 3, 1) #the input needs to be RGB
            labels = labels.long()
            emb, outputs = model(images) #at index 1 there is the output of the last layer while at 0 the embeddings after the pooling layer
            #emb is [batch_size, 2048, 1, 1], it is reshaped to [batch_size, 2048] and then concatenated with radiomix_feats so as to have a tensor of shape [batch_size, 2048+190]
            input_FCNN = torch.cat((emb.reshape(emb.shape[0], emb.shape[1]), radiomix.float()), dim=1)
            outputs = FCNN_model(input_FCNN)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item()
            epoch_acc += torch.sum(preds == labels.data)
        epoch_loss /= len(val_loader.dataset)
        epoch_acc = epoch_acc.double() / len(val_loader.dataset)
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        print('Epoch:', epoch, 'Val loss:', epoch_loss, 'Val acc:', epoch_acc)
    return model, train_loss, val_loss, train_acc, val_acc

model, train_loss, val_loss, train_acc, val_acc = train_model(resnet_model, train_dataset, val_dataset, batch_size=8, num_epochs=10, learning_rate=0.001)
#give a resume of the training
if PLOT:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#save the model in the same working dir of this file
torch.save(model.state_dict(), 'model.pth')

#load the trained model just saved
model = resnet50(pretrained=False, num_classes=2)
model.load_state_dict(torch.load('model.pth'))

#test the model on the test set and do other stuff


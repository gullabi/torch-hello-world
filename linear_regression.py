import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# define class for linear regression
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

# define train test split function
def ttss(X,y,ratio_train):
    
    full_data = np.hstack([X,y])
    np.random.shuffle(full_data)

    train = full_data[:round(np.shape(full_data)[0]*ratio_train),:]
    test = full_data[round(np.shape(full_data)[0]*ratio_train):,:]
        
    return train[:,:-1], test[:,:-1], np.expand_dims(train[:,-1], axis=1), np.expand_dims(test[:,-1], axis=1) # Xtrain, Xtest, ytrain, ytest
    

# read in data from prepared_data.csv
with open('prepared_data.csv') as f:
    Xy = np.loadtxt(f, delimiter=',')

inputs = Xy[:,[1]].astype(np.float32) # pick one column from data
targets = np.expand_dims(Xy[:,-1], axis=1).astype(np.float32) # last column is target

train_ratio = .8 
inputDim = np.shape(inputs)[1]
outputDim = 1
learningRate = 0.000001
epochs = 10

X_train, X_test, y_train, y_test = ttss(inputs, targets, train_ratio)
print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

# inputs = torch.from_numpy(Xy[:,:-2])
# targets = torch.from_numpy(Xy[:,-1])
# dataset = TensorDataset(inputs, targets)
# batch_size = 3
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# instantate the model
model = linearRegression(inputDim, outputDim)

# For GPU
if torch.cuda.is_available():
    model.cuda()

# define MSE as the learning rule
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# start training
for epoch in range(epochs):
    # converting inputs and labels to torch variables
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(X_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())

    else:
        inputs = Variable(torch.from_numpy(X_train))
        labels = Variable(torch.from_numpy(y_train))

    # clear gradient buffers
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # calculate MSE loss using criterion
    loss = criterion(outputs, labels)

    # calculate gradients of the loss function with respect to the parameters
    loss.backward()

    #update the parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            print('bias: ',layer.state_dict()['bias'] ,' weight(s): ', layer.state_dict()['weight'])


with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(X_test).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(X_test))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(X_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(X_test, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
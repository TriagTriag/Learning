# including mushroom dataset

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

# read data
dataset = 'adult'
if (dataset=='mushroom'):
    df = pd.read_csv('mushrooms.csv')
elif (dataset=='adult'):
    df = pd.read_csv('adult.csv')



# split data
if (dataset=='mushroom'):
    # convert to numerical data
    df = pd.get_dummies(df)
    X = df.drop({'class_p', 'class_e'}, axis=1)
    y = df['class_p']
elif (dataset=='adult'):
    # clean data
    df = df[df['workclass'] != ' ?']
    df = df[df['occupation'] != ' ?']
    df = df[df['native-country'] != ' ?']

    # convert to numerical data
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop({'income_>50K', 'fnlwgt'}, axis=1)
    y = df['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model with height=3
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# test model
print(clf.score(X_test, y_test))

# find the prediction of the model
y_pred_h = clf.predict(X_test)

# plot tree
plt.figure(figsize=(20, 20))
tree.plot_tree(clf, feature_names=X.columns, class_names=['edible', 'poisonous'], filled=True)
plt.show()

# convert to tensor
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# train a neural network

class FCs(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCs, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# train model
model = FCs(X_train.shape[1], 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
losses = []

for i in range(epochs):
    i += 1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 1:
        print(f'epoch: {i:3} loss: {loss.item():10.8f}')
        # print accuracy
        with torch.no_grad():
            y_eval = model.forward(X_test)
            # softmax of y_eval
            y_eval = torch.sigmoid(y_eval)
            acc = y_eval.round().eq(y_test).sum() / float(y_test.shape[0])
            print(f'accuracy: {acc:.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# convert losses to numpy array
losses_np = np.zeros(epochs)
for i in range(len(losses)):
    losses_np[i] = losses[i].detach().numpy()

# plot loss
plt.plot(range(epochs), losses_np)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

# test model and find accuracy
with torch.no_grad():
    y_eval = model.forward(X_test)
    # softmax of y_eval
    y_eval = torch.sigmoid(y_eval)
    acc = y_eval.round().eq(y_test).sum() / float(y_test.shape[0])

print(f'accuracy: {acc:.8f}')


# find the instances that y_eval is wrong and clf is right
h_false_a_true = []
for i in range(len(y_eval)):
    if y_eval[i].round() != y_test[i] and y_pred_h[i] == y_test[i]:
        h_false_a_true.append(i)

# find the instances that y_eval is right and clf is wrong
h_true_a_false = []
for i in range(len(y_eval)):
    if y_eval[i].round() == y_test[i] and y_pred_h[i] != y_test[i]:
        h_true_a_false.append(i)

# find the instances that y_eval is wrong and clf is wrong
h_false_a_false = []
for i in range(len(y_eval)):
    if y_eval[i].round() != y_test[i] and y_pred_h[i] != y_test[i]:
        h_false_a_false.append(i)

# find the instances that y_eval is right and clf is right
h_true_a_true = []
for i in range(len(y_eval)):
    if y_eval[i].round() == y_test[i] and y_pred_h[i] == y_test[i]:
        h_true_a_true.append(i)

# print the proportion of each case
print(f'Human is right and classifier is wrong: {len(h_false_a_true) / len(y_eval)}')
print(f'Human is wrong and classifier is right: {len(h_true_a_false) / len(y_eval)}')
print(f'Human is wrong and classifier is wrong: {len(h_false_a_false) / len(y_eval)}')
print(f'Human is right and classifier is right: {len(h_true_a_true) / len(y_eval)}')


# save X_train, Y_train, clf(X_train), model(X_train), X_test, Y_test, clf(X_test), model(X_test)
# to a npz file
y_pred_class_train = torch.sigmoid(model.forward(X_train)).round().detach().numpy()
# find minimum of sigmoid and 1-sigmoid
# loss_pred_class_train = torch.min(torch.sigmoid(model.forward(X_train)), 1 - torch.sigmoid(model.forward(X_train))).detach().numpy()
y_pred_class_test = torch.sigmoid(model.forward(X_test)).round().detach().numpy()
# loss_pred_class_test = torch.min(torch.sigmoid(model.forward(X_test)), 1 - torch.sigmoid(model.forward(X_test))).detach().numpy()
# convert to numpy array
X_train = X_train.numpy()
X_test = X_test.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()
y_pred_h_train = clf.predict(X_train).reshape(-1, 1)
y_pred_h_test = clf.predict(X_test).reshape(-1, 1)

# save to npz file
if (dataset == 'mushroom'):
    np.savez('mushroom.npz', X_train=X_train, y_train=y_train, y_pred_h_train=y_pred_h_train, y_pred_class_train=y_pred_class_train, X_test=X_test, y_test=y_test, y_pred_h_test=y_pred_h_test, y_pred_class_test=y_pred_class_test)
if (dataset == 'adult'):
    np.savez('adult.npz', X_train=X_train, y_train=y_train, y_pred_h_train=y_pred_h_train, y_pred_class_train=y_pred_class_train, X_test=X_test, y_test=y_test, y_pred_h_test=y_pred_h_test, y_pred_class_test=y_pred_class_test)

import numpy as np
import torch
from torch import nn
import torch as T

from neural_network import NeuralNetwork
from people_data_set import PeopleDataset
from sklearn.preprocessing import MinMaxScaler


def main():
    device = T.device("cpu")  # to Tensor or Module
    print("\nBegin PyTorch DataLoader demo ")

    # 0. miscellaneous prep
    T.manual_seed(0)
    np.random.seed(0)

    print("\nSource data looks like: ")
    print("1 0  0.171429  1 0 0  0.966805  0")
    print("0 1  0.085714  0 1 0  0.188797  1")
    print(" . . . ")

    # 1. create Dataset and DataLoader object
    print("\nCreating Dataset and DataLoader ")

    train_file = "E:\\AI\\AI\\SimpleAI\\people.txt"
    train_dataset_with_input_and_output = PeopleDataset(train_file, device, num_rows=8)

    bat_size = 3
    # Training Data Sample Size
    training_dataset_data_loader = T.utils.data.DataLoader(train_dataset_with_input_and_output,
                                        batch_size=bat_size, shuffle=True)

    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork(7,3).to(device)
    print(model)


    # 2. iterate thru training data twice
    for epoch in range(2):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))
        for (batch_idx, batch) in enumerate(training_dataset_data_loader):
            print("\nBatch = " + str(batch_idx))
            inputs = batch['predictors']  # [3,7]
            # Y = T.flatten(batch['political'])  #
            outputs = batch['political']  # [3]
            print(inputs)
            print(outputs)
            logits = model(inputs)
            print(logits)

            scalar = MinMaxScaler((0, 2))
            pred_probab = scalar.fit_transform(logits.data)
            pred_probab = torch.max(torch.tensor(pred_probab), 1)
            print(pred_probab.values)
            #y_pred = pred_probab.argmax(1)
            #print(f"Predicted class: {y_pred}")
    print("\n==============================")

    print("\nEnd demo ")


if __name__ == "__main__":
    main()

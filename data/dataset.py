import torch

# Anomaly is only present in testing data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, time, stable, label, window_size, multi_entity=False):
        self.data = data
        self.time = time
        self.stable = stable
        self.label = label
        self.window_size = window_size

    def __getitem__(self, index):
        # if self.multi_entity:
        #     data = self.data[index]
        #     time = self.time[index]
        #     stable = self.stable[index]
        #     label = self.label[index]    
        
        data = self.data[index: index + self.window_size, :]
        time = self.time[index: index + self.window_size, :]
        stable = self.stable[index: index + self.window_size, :]
        label = self.label[index: index + self.window_size, :]

        return data, time, stable, label

    def __len__(self):
        return len(self.data) - self.window_size + 1

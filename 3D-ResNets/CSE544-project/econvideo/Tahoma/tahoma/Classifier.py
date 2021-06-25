import torch.nn.functional as F

class Classifier:
    def __init__(self, model_list, low_high_list):
        self.model_list = model_list
        self.low_high_list = low_high_list
        for i in range(len(self.model_list)):
            self.model_list[i].eval()
            
    def __call__(self, images):
        return self.forward(images)

    def forward(self, images):
        # images have sizes corresponding to the models in the cascade
        for i in range(len(self.model_list) - 1):
            # print(self.low_high_list[i], i)
            output = self.model_list[i](images[i])
            # print('output', output.data)
            prediction = F.softmax(output.data, dim=1)[0, 1]
            # print('prediction', prediction.item())
            if prediction.item() <= self.low_high_list[i][0]:
                return 0
            elif prediction.item() >= self.low_high_list[i][1]:
                return 1
        output = self.model_list[-1](images[len(self.model_list) - 1])
        # print('output', output.data)
        prediction = F.softmax(output.data, dim=1)[0, 1]
        # print('prediction', prediction.item())
        return 1 if prediction.item() > 0.5 else 0
        
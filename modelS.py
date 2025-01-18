import torch
import torch.nn as nn
import torchvision.models as models

class SiameseResNet(nn.Module):
    def __init__(self, model_path=None):
        super(SiameseResNet, self).__init__()

        if model_path:
            if 'resnet50' in model_path:
                model = models.resnet50()
            else:
                model = models.resnet101()
            model.fc = nn.Linear(model.fc.in_features, 700)
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward_one(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # 展平
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

    def predict(self, query_image, support_features):
        query_feature = self.forward_one(query_image.unsqueeze(0))
        distances = []
        for feature, label in support_features:
            distance = torch.nn.functional.pairwise_distance(query_feature, feature)
            distances.append((distance.item(), label))
        # 找到最近的支持样本的类别
        pred = min(distances, key=lambda x: x[0])[1]
        return pred
    
    def predict_prototypical(self, query_image, support_features):
        query_feature = self.forward_one(query_image.unsqueeze(0))
        label_to_features = {}
        for feature, label in support_features:
            if label not in label_to_features:
                label_to_features[label] = []
            label_to_features[label].append(feature)

        min_dist = float('inf')
        pred = list(label_to_features.keys())[0]
        for label in label_to_features.keys():
            feature = torch.stack(label_to_features[label]).mean(dim=0)
            distance = torch.nn.functional.pairwise_distance(query_feature, feature)
            if distance < min_dist:
                min_dist = distance
                pred = label

        return pred

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)  

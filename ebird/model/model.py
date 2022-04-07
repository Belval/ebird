import torch
from ebird.model.backbone import get_backbone

class Model(torch.nn.Module):
    def __init__(self, config):
        """
        Initialize our model from the model config defined in the YAML file
        """
        super(Model, self).__init__()
        self.backbone = get_backbone(config["BACKBONE"])

        if config["CLASSIFICATION_HEAD"]["TYPE"] == "mlp":
            self.backbone.fc = torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"] - 80, config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"] - 80)
            self.fc = torch.nn.Sequential(
                *([
                    torch.nn.Sequential(
                        torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"]),
                        torch.nn.ReLU()
                    )
                    for _ in range(config["CLASSIFICATION_HEAD"].get("LAYER_COUNT", 5) - 1)
                ])
            )
            self.label_prediction = torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], config["CLASSIFICATION_HEAD"]["OUTPUT_FEATURES"])
            self.label_count = torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], 1)
        else:
            raise NotImplementedError()

    def forward(self, input_images, input_features):
        outputs = self.backbone(input_images)

        outputs = self.fc(torch.hstack((outputs, input_features)).to(torch.float32))
        labels = self.label_prediction(outputs)
        label_count = self.label_count(outputs)

        return labels, label_count

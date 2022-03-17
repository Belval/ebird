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
            self.backbone.fc = torch.nn.Linear(2048, 2048)
            self.classification_head = torch.nn.Sequential(
                torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"]),
                torch.nn.ReLU(),
                torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"]),
                torch.nn.ReLU(),
                torch.nn.Linear(config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"], config["CLASSIFICATION_HEAD"]["OUTPUT_FEATURES"])
            )
        else:
            self.classification_head = None
            self.backbone.fc = torch.nn.Linear(
                config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"],
                config["CLASSIFICATION_HEAD"]["OUTPUT_FEATURES"],
            )

    def forward(self, input_images, input_features):
        outputs = self.backbone(input_images)

        if self.classification_head is not None:
            outputs = self.classification_head(torch.hstack((outputs, input_features)).to(torch.float32))

        return outputs

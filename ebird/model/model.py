import torch
from ebird.model.backbone import get_backbone

class Model(torch.nn.Module):
    def __init__(self, config):
        """
        Initialize our model from the model config defined in the YAML file
        """
        super(Model, self).__init__()
        self.backbone = get_backbone(config["BACKBONE"])
        self.backbone.fc = torch.nn.Linear(
            config["CLASSIFICATION_HEAD"]["INPUT_FEATURES"],
            config["CLASSIFICATION_HEAD"]["OUTPUT_FEATURES"],
        )

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        #outputs = self.classification_head(inputs)

        return outputs

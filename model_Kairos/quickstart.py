import torch
from tsfm.model.kairos import AutoModel

# load model
model = AutoModel.from_pretrained(
    "mldi-lab/Kairos_50m", trust_remote_code=True
)

# forecasting configurations
batch_size, context_length, prediction_length = 1, 2048, 96
seqs = torch.randn(batch_size, context_length)

prediction_length = 96
forecast = model(
    past_target=seqs.clone().detach().float(),
    prediction_length=prediction_length,
    generation=True,
    preserve_positivity=True,
    average_with_flipped_input=True
)

# extract the prediction results
forecast = forecast["prediction_outputs"]
print(forecast.shape)
print(forecast)
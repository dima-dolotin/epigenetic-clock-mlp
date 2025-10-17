import torch
import mlflow
import mlflow.pytorch

model_path = './model'   # folder containing MLmodel
inference_set = 'testset.pt'

# load model
model = mlflow.pytorch.load_model(model_path)

model.eval()

data = torch.load(inference_set, map_location='cpu')
x = data['test_inputs']
y = data['test_labels']

# run inference
with torch.no_grad():
    y_pred = model(x)

print('Prediction:', y_pred)
print('Actual:', y)

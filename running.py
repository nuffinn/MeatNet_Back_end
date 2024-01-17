import torch
from src.evaluation.missclassification_cost import evaluate_mcc
import pandas as pd

# Read the CSV file into a DataFrame
csv_path = '~/Phteven/models/resnet18fe_test_predict.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_path)

# Extract the 'predict', 'true', and 'filename' columns as tensors
predict_tensor = torch.tensor(df['predict'].values)
true_tensor = torch.tensor(df['label'].values)

# Convert numerical values to one-hot encoded tensors with swapped indices
num_classes = 3  # Assuming there are three classes (0, 1, 2)
predict_one_hot = torch.eye(num_classes)[2 - predict_tensor.long()]
true_one_hot = torch.eye(num_classes)[2 - true_tensor.long()]

# Call the evaluate_mcc function with one-hot encoded tensors
result = evaluate_mcc(predict_one_hot, true_one_hot)

# Print the result
print("Final Cost:", result)

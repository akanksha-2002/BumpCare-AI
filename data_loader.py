import kagglehub

# Download latest version of dataset
path = kagglehub.dataset_download("csafrit2/maternal-health-risk-data")

print("Path to dataset files:", path)

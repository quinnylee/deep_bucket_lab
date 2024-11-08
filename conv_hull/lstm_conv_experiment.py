# Import packages
import sys
import yaml
import torch
import os
import multiprocessing as mp

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)


from data_generation import BucketSimulation
from model_controller import ModelController
from validation import ModelValidator
from scipy.spatial import ConvexHull
import numpy as np
from scipy.optimize import linprog
import pandas as pd
from scipy.stats import mannwhitneyu

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('../configuration/configuration.yml')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
print("beginning", flush=True)

# Initialize and generate synthetic data for each split
bucket_sim_train = BucketSimulation(config, 'train')
bucket_sim_val = BucketSimulation(config, 'val')
#bucket_sim_test = BucketSimulation(config, 'test')
print("initialization done", flush=True)

# Simulate and store data for training, validation, and testing
train_data = bucket_sim_train.generate_data(config['synthetic_data']['train']['num_records'])
val_data = bucket_sim_val.generate_data(config['synthetic_data']['val']['num_records'])
#test_data = bucket_sim_test.generate_data(config['synthetic_data']['test']['num_records'])
print("simulation done", flush=True)

bucket_dictionary = {
    'train': train_data,
    'val': val_data
    #'test': test_data
}

# linear programming approach to determine if a point y lies within a conv hull defined by 
# points or not
def in_hull(points, y):
    n_points = len(points)
    n_dim = len(y)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[y.T, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

features_of_interest = ['precip', 'H_bucket', 'rA_spigot', 'rH_spigot'] # can be changed

# make list of training and val points 
train_points = bucket_dictionary['train'][features_of_interest] 
train_points = train_points.to_numpy()
# points in the first sequence don't get predicted on, so we have to truncate our val periods
val_all = bucket_dictionary['val'][features_of_interest] 
val_all = val_all.to_numpy()
val_points = []

# Ignores points in the first sequence passed into the LSTM, since they don't get predicted
# In this setup, we are only selecting one point out of every bucket to ensure that samples
# are independent of one another.
for i in range(len(val_all)):
    if (i % (config['model']['seq_length'] + 1) == config['model']['seq_length']):
        val_points.append(val_all[i])

val_points = np.array(val_points)


# generate convex hull vertices to reduce computational time for in_hull
hull = ConvexHull(train_points)
vertices = train_points[hull.vertices]
print("conv hull done", flush=True)

# Parallel processing to test if points are in conv hull or not
def process_test_point(test_point):
    return in_hull(vertices, test_point)

with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.map(process_test_point, val_points)

numInterp = sum(results)
numExterp = len(results) - numInterp

# Print properties of data
print("Number of interpolations: ", numInterp)
print("Number of extrapolations: ", numExterp)
print("Number of training buckets: ", config['synthetic_data']['train']['n_buckets'])
print("Number of val buckets: ", config['synthetic_data']['val']['n_buckets'])
print("Number of training points per bucket: ", str(config['synthetic_data']['train']['num_records'] - config['warmup_period']))
print("Number of val points per bucket: ", str(config['synthetic_data']['val']['num_records'] - config['warmup_period'] - config['model']['seq_length']))
print('cateogorization done', flush=True)

# Initialize the LSTM model and Model Controller
model_controller = ModelController(config, device, bucket_dictionary)

# Prepare data loaders
train_loader = model_controller.make_data_loader('train')
val_loader = model_controller.make_data_loader('val')
#test_loader = model_controller.make_data_loader('test')

# Now train_loader, val_loader, and test_loader should be dictionaries
print("training model", flush=True)
trained_model = model_controller.train_model(train_loader)

model_validator = ModelValidator(trained_model, device, 
                                 bucket_dictionary, val_loader, 
                                 config, "val", model_controller.scaler_out)
print("validating model", flush=True)
losses = model_validator.validate_model(get_losses=True)
print(losses)

print("results length, ", len(results))
print("losses length ", len(losses))

categories = pd.DataFrame(
    {'result': results,
    'loss': losses}
)

# Categorize losses into interpolated or extrapolated
interpolated = categories[categories['result'] == True]
extrapolated = categories[categories['result'] == False]

print(interpolated['loss'])
print('\n')
print(extrapolated['loss'])
'''
mean_interp = interpolated['loss'].mean()
std_interp = interpolated['loss'].std()
mean_extrap = extrapolated['loss'].mean()
std_extrap = extrapolated['loss'].std() 

# Mann-Whitney U stats test for data without normal dists
stat, p_value = mannwhitneyu(interpolated['loss'], extrapolated['loss'], alternative='less')
print("MW U-statistic: ", stat)
print("p-value: ", p_value)'''
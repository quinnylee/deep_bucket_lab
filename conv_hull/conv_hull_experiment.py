import sys
sys.path.append('./src/')
import yaml
import torch
from data_generation import BucketSimulation
from model_controller import ModelController
from validation import ModelValidator
from scipy.spatial import ConvexHull

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('./configuration/configuration.yml')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')

# Initialize and generate synthetic data for each split
bucket_sim_train = BucketSimulation(config, 'train')
bucket_sim_val = BucketSimulation(config, 'val')
bucket_sim_test = BucketSimulation(config, 'test')

# Simulate and store data for training, validation, and testing
train_data = bucket_sim_train.generate_data(config['synthetic_data']['train']['num_records'])
val_data = bucket_sim_val.generate_data(config['synthetic_data']['val']['num_records'])
test_data = bucket_sim_test.generate_data(config['synthetic_data']['test']['num_records'])

bucket_dictionary = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}


def in_hull(points, y):
    n_points = len(points)
    n_dim = len(y)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[y.T, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

features_of_interest = ['precip', 'H_bucket', 'rA_spigot', 'rH_spigot'] # can be changed
train_points = bucket_dictionary['train']
test_points = bucket_dictionary['test']

results = []
dists = []
hull = ConvexHull(train_points)
vertices = train_points[hull.vertices]

for i in range(len(test_points)):
    results.append(in_hull(train_points, test_points[i]))
    
    if results[-1]:
        dists.append(0)
    else:
        X = vertices # points in training set
        P = np.dot(X, X.T)  # quick way to build a symmetric matrix
        ext_pt = np.array(test_points[i])
        q = np.dot(X, ext_pt.T).T * (-1) # multiply ext pt with train polytope
        A = np.ones((1,len(X)))
        b = np.array([1.])
        lb = np.zeros((1,len(X)))
        x = solve_qp(P, q, A=A, b=b, lb=lb, solver="osqp")

        nearest_pt = np.dot(x, X)
        dist = np.linalg.norm((ext_pt - nearest_pt)) 

        dists.append(dist) 
            
print(results)
numInterp = 0
numExterp = 0

for result in results:
    if result:
        numInterp += 1
    else:
        numExterp += 1

print("Number of interpolations: ", numInterp)
print("Number of extrapolations: ", numExterp)
print("Number of training buckets: ", n_buckets_split["train"])
print("Number of testing buckets: ", t_n_buckets_split["test"])
print("Number of training points per bucket: ", time_splits["train"])
print("Number of testing points per bucket: ", t_time_splits["test"])
print('done')



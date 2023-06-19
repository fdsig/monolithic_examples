# Import necessary modules from the Azure Machine Learning Python SDK
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
import os
# Create a Python environment for the experiment.
# The environment encapsulates the dependencies needed for the script to run.
train_env = Environment("train-env")
# Set to False because Azure ML will manage dependencies for us.
train_env.python.user_managed_dependencies = False 

# Specify Docker steps as a string.
# Dockerfile describes the base image and additional steps needed to create an environment suitable for running the script.
# This Dockerfile starts with a preconfigured Azure ML image, sets some environment variables, and prints a message.
dockerfile = f"""
FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
ENV WANDB_API={os.environ['WANDB_API']}
ENV WANDB_ENTITY=demonstrations
ENV WANDB_PROJECT=from_azure_lightning
RUN echo "Hello from custom container!"
"""

# Set the base image to None, because the image is defined by Dockerfile.
train_env.docker.base_image = None
# Pass the Dockerfile to the environment.
train_env.docker.base_dockerfile = dockerfile

# Create a set of package dependencies.
# These are the Python packages needed by the script.
train_dependencies = CondaDependencies.create(conda_packages=['numpy'],
                                               pip_packages=['wandb', 'azureml-sdk','torch','pytorch-lightning','torchvision'])

# Add the dependencies to the environment.
train_env.python.conda_dependencies = train_dependencies

# Initialize a Workspace object from the existing workspace defined in the configuration file.
ws = Workspace.from_config()

# Choose a name for your cluster.
cluster_name = "test-cluster_2"

# Try to get the compute target (i.e., the remote Azure Machine Learning Compute resource to run your training script on)
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D2_v2', max_nodes=1)

    # Create the cluster with the specified name and configuration
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # Wait for the cluster to be provisioned, show the output log
    compute_target.wait_for_completion(show_output=True)

# Get a detailed status for the current AmlCompute target and print it.
print(compute_target.get_status().serialize())

# Configure the training job
# - Specify the directory that contains your scripts. All the files in this directory are uploaded into the cluster nodes for execution. 
# - Specify the compute target to execute the script on.
# - Specify the training script to run.
src = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      compute_target=compute_target,
                      environment=train_env)

# Submit the run to the experiment.
run = Experiment(ws,'Tutorial-train').submit(src)

# Show the running experiment run in the notebook widget
# It shows a link to view the run in the Azure portal, a link to stream logs, and a summary of the run status.
run.wait_for_completion(show_output=True)

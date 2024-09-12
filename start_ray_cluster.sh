#!/bin/bash

# Number of worker nodes to start
num_nodes=$1

# Start Ray head node and capture the output
output=$(ray start --head --port=6379)

# Extract the address and password
address=$(echo "$output" | grep -o -P '(?<=--address=)[^ ]+')
redis_password=$(echo "$output" | grep -o -P '(?<=--redis-password=)[^ ]+')

# Print the extracted address and password
echo "Address: $address"
echo "Redis Password: $redis_password"

# Function to submit a single job
submit_job() {
  sbatch <<EOL
#!/bin/bash
#SBATCH --job-name=ray-worker
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --mem=180GB

# Function to stop Ray and kill Python processes run by 'carmon' when the script exits
cleanup() {
  ray stop
  pkill -f python -u carmon
}
trap cleanup EXIT

# Activate the Conda environment with Ray installed
source /beegfs/store/carmon/anaconda3/etc/profile.d/conda.sh
conda activate Dreams2

# Command to connect to the head node
ray start --address='$address' --redis-password='$redis_password'
sleep 3600
EOL
}

# Submit the specified number of worker jobs
for ((i=0; i<$num_nodes; i++))
do
  submit_job
done

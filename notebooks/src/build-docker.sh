# The name of our algorithm
algorithm_name=ezsmdeploy-image-$1

echo "Building container ${algorithm_name}"

cd src 

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}"

echo "Building ${fullname}"

echo "Creating repo for ${fullname} if it doesn't already exist"
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

echo "Getting login for ${fullname}"
# Get the login command from ECR and execute it directly
aws --region ${region} ecr get-login-password | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

echo "Building locally"
docker build -q -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

echo "Pushing"
docker push ${fullname}
 
echo "${fullname}"

# Have to do this because pythons subprocess for calling this script does not wait for it to finish. Tried various args to Popen
sudo touch done.txt

echo "SUCCESS"
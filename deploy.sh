# deploy.sh
# Automates the Build -> Deploy process

echo "=== 1. Building Docker Image ==="
docker build -t job-role-predictor:latest .

echo "\n=== 2. Applying Kubernetes Manifests ==="
# Check if kubectl is available
if ! command -v kubectl &> /dev/null
then
    echo "kubectl could not be found. Please ensure it is installed."
    exit 1
fi

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

echo "\n=== 3. Verifying Rollout Status ==="
kubectl rollout status deployment/job-role-predictor-deploy

echo "\n=== Deployment Complete ==="
echo "Access the service at the LoadBalancer IP or NodePort."

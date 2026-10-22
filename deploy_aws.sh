#!/usr/bin/env bash
# Deploys (or updates) the WaxFashionStyleGAN API on AWS Lambda as a container
# image with a public function URL. Run it in AWS CloudShell, which already has
# the AWS CLI and Docker:
#
#   HF_TOKEN=hf_xxx ./deploy_aws.sh
#
# Optional: AWS_REGION (default us-east-1), FUNCTION, REPO, SOURCE_IMAGE,
# MEMORY_MB (default 10240 = 6 vCPU), MAX_CONCURRENCY (default 5, a hard cap on
# cost), ALLOWED_ORIGINS (e.g. https://waxfashion-web-frontend.vercel.app),
# ROLE_ARN (an existing Lambda execution role, e.g. LabRole in AWS Academy).
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
FUNCTION=${FUNCTION:-waxfashion-api}
REPO=${REPO:-waxfashion-api}
SOURCE_IMAGE=${SOURCE_IMAGE:-ghcr.io/anelbazarbayeva95/waxfashion-api:latest}
MEMORY_MB=${MEMORY_MB:-10240}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-5}
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}
HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token that can read the model}
ROLE_ARN=${ROLE_ARN:-}

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPO:latest"
export AWS_DEFAULT_REGION=$REGION

# ── 1. Copy the image from GitHub's registry to ECR (Lambda only runs ECR images)
echo "1/4 Copying the backend image to Amazon ECR..."
aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
docker pull --platform linux/amd64 "$SOURCE_IMAGE"
docker tag "$SOURCE_IMAGE" "$IMAGE"
docker push "$IMAGE"
docker image rm "$SOURCE_IMAGE" "$IMAGE" >/dev/null || true   # CloudShell disk is small

# ── 2. Execution role ────────────────────────────────────────────────────────
echo "2/4 Setting up the Lambda execution role..."
if [ -z "$ROLE_ARN" ]; then
  ROLE_NAME="$FUNCTION-role"
  if ! ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text 2>/dev/null); then
    TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    if ROLE_ARN=$(aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" \
                    --query Role.Arn --output text 2>/dev/null); then
      aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      sleep 10  # a new role takes a few seconds before Lambda can use it
    elif ROLE_ARN=$(aws iam get-role --role-name LabRole --query Role.Arn --output text 2>/dev/null); then
      echo "    Can't create roles here; using the existing LabRole."
    else
      echo "Couldn't create an IAM role. Rerun with ROLE_ARN=<an existing Lambda role ARN>." >&2
      exit 1
    fi
  fi
fi

# ── 3. Create or update the function ─────────────────────────────────────────
echo "3/4 Deploying the Lambda function..."
ENV_JSON=$(printf '{"Variables":{"HF_TOKEN":"%s","ALLOWED_ORIGINS":"%s"}}' "$HF_TOKEN" "$ALLOWED_ORIGINS")
# 300 s timeout covers a cold start (model download + load); 2 GB of /tmp holds the 369 MB model.
if aws lambda get-function --function-name "$FUNCTION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION"
  aws lambda update-function-configuration --function-name "$FUNCTION" \
    --memory-size "$MEMORY_MB" --timeout 300 --ephemeral-storage Size=2048 \
    --environment "$ENV_JSON" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION"
else
  aws lambda create-function --function-name "$FUNCTION" --package-type Image \
    --code ImageUri="$IMAGE" --role "$ROLE_ARN" --architectures x86_64 \
    --memory-size "$MEMORY_MB" --timeout 300 --ephemeral-storage Size=2048 \
    --environment "$ENV_JSON" >/dev/null
  aws lambda wait function-active-v2 --function-name "$FUNCTION"
fi

# At most MAX_CONCURRENCY copies run at once: a hard ceiling on cost
aws lambda put-function-concurrency --function-name "$FUNCTION" \
  --reserved-concurrent-executions "$MAX_CONCURRENCY" >/dev/null \
  || echo "    Note: couldn't set the concurrency cap (new accounts have a low overall limit). The function still works."

# ── 4. Public HTTPS URL ──────────────────────────────────────────────────────
echo "4/4 Setting up the public URL..."
if ! URL=$(aws lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text 2>/dev/null); then
  URL=$(aws lambda create-function-url-config --function-name "$FUNCTION" --auth-type NONE \
          --query FunctionUrl --output text)
  # Public access needs both permissions; the second is limited to calls made through the URL
  aws lambda add-permission --function-name "$FUNCTION" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null
fi
URL=${URL%/}

echo
echo "Backend URL: $URL"
echo "Health:      $URL/health   (first call starts the function and loads the model, ~30-60 s)"
echo "Test image:  $URL/image?seed=137&psi=0.7"
echo
echo "Put $URL in frontend/config.js, then push; Vercel redeploys the site."

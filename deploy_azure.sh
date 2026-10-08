#!/usr/bin/env bash
# Deploys (or updates) the WaxFashionStyleGAN API on Azure Container Apps.
#
#   az login
#   HF_TOKEN=hf_xxx IMAGE=ghcr.io/<owner>/waxfashion-api:latest ./deploy_azure.sh
#
# Optional: RG, LOCATION, ENV_NAME, APP, ALLOWED_ORIGINS (comma-separated,
# e.g. https://waxfashion.vercel.app), MIN_REPLICAS, MAX_REPLICAS.
set -euo pipefail

RG=${RG:-waxfashion-rg}
LOCATION=${LOCATION:-eastus}
ENV_NAME=${ENV_NAME:-waxfashion-env}
APP=${APP:-waxfashion-api}
IMAGE=${IMAGE:?Set IMAGE, e.g. ghcr.io/<owner>/waxfashion-api:latest}
HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token that can read the model}
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}
MIN_REPLICAS=${MIN_REPLICAS:-0}   # 0 = scale to zero when idle (cheapest)
MAX_REPLICAS=${MAX_REPLICAS:-3}

ENV_VARS=(HF_TOKEN=secretref:hf-token "ALLOWED_ORIGINS=$ALLOWED_ORIGINS" TORCH_THREADS=2)

az extension add --name containerapp --upgrade -y --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group create -n "$RG" -l "$LOCATION" -o none

if ! az containerapp env show -n "$ENV_NAME" -g "$RG" -o none 2>/dev/null; then
  echo "Creating Container Apps environment (takes a few minutes)..."
  az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" -o none
fi

if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  echo "Updating $APP..."
  az containerapp secret set -n "$APP" -g "$RG" --secrets "hf-token=$HF_TOKEN" -o none
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" \
    --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
    --set-env-vars "${ENV_VARS[@]}" -o none
  # Running replicas don't pick up a changed secret until they restart
  REVISION=$(az containerapp show -n "$APP" -g "$RG" --query properties.latestRevisionName -o tsv)
  az containerapp revision restart -n "$APP" -g "$RG" --revision "$REVISION" -o none
else
  echo "Creating $APP..."
  # 2 vCPU / 4 GiB per replica. Each replica runs one image at a time, so
  # scale out once 2 requests are waiting on a replica.
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --image "$IMAGE" --target-port 8000 --ingress external \
    --cpu 2 --memory 4Gi \
    --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
    --scale-rule-name http-load --scale-rule-type http --scale-rule-http-concurrency 2 \
    --secrets "hf-token=$HF_TOKEN" \
    --env-vars "${ENV_VARS[@]}" -o none
fi

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "Backend URL: https://$FQDN"
echo "Health:      https://$FQDN/health   (first call after idle starts the container, ~30-60 s)"
echo "Test image:  https://$FQDN/image?seed=137&psi=0.7"
echo
echo "Put https://$FQDN in frontend/config.js, then deploy the frontend to Vercel."

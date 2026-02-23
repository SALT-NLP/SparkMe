#!/bin/bash

# =============================================================================
# Quick Deploy Script for Annotation App to GCP Cloud Run
# =============================================================================

# TODO: CONFIGURATION (fill this)
PROJECT_ID="..."
SERVICE_NAME="..."
REGION="..."
BUCKET_NAME="...-${PROJECT_ID}"

# TODO: Environment variables for Cloud Run (fill this)
NON_SENSITIVE_VARS=""
NON_SENSITIVE_VARS+="USERS_FILE=...," # your users file storing login information
NON_SENSITIVE_VARS+="INTERVIEW_DATA_ROOT=...,"
NON_SENSITIVE_VARS+="ANNOTATIONS_ROOT=...," # annotations folder

echo "================================================"
echo "Deploying Annotation App to Cloud Run"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo "================================================"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# Bucket already exists from interview app deployment
echo "Using existing bucket: gs://$BUCKET_NAME"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars "$NON_SENSITIVE_VARS" \
  --set-secrets "SECRET_KEY=flask-secret-key:latest" \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET_NAME \
  --add-volume-mount volume=data,mount-path=/app/data

# Get the URL
echo ""
echo "================================================"
echo "Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo ""
echo "Annotation App URL:"
echo "   $SERVICE_URL"
echo ""
echo "Annotations will be saved to:"
echo "   gs://$BUCKET_NAME/annotations/annotations.csv"
echo ""
echo "================================================"
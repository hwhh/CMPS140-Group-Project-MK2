BUCKET_NAME=bucket1
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil -m cp -r input gs://$BUCKET_NAME/input

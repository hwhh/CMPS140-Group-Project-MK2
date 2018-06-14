OUTPUT_PATH=gs://$BUCKET_NAME/mfcc_single/logs/
TRAIN_DIR=gs://$BUCKET_NAME/input/audio_train/
TEST_DIR=gs://$BUCKET_NAME/input/audio_test/
TEST_JSON=gs://$BUCKET_NAME/input/test.json
TRAIN_DATA=gs://$BUCKET_NAME/input/train.csv
EVAL_DATA=gs://$BUCKET_NAME/input/sample_submission.csv
REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer_1/ \
    --region $REGION \
    --config=config.yaml \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --model_level 3 \
    --n_mfcc 40 \
    --audio_duration 2 \
    --user_arg_1 $TRAIN_DIR \
    --user_arg_2 $TEST_DIR \
    --learning-rate 0.001 \
    --verbosity DEBUG

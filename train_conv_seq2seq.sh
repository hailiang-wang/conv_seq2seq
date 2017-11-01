export PYTHONIOENCODING=UTF-8

export DATA_PATH=data/textsum_all

export VOCAB_SOURCE=${DATA_PATH}/vocab.x
export VOCAB_TARGET=${DATA_PATH}/vocab.y
export TRAIN_SOURCES=${DATA_PATH}/train.x
export TRAIN_TARGETS=${DATA_PATH}/train.y
export DEV_SOURCES=${DATA_PATH}/valid.x
export DEV_TARGETS=${DATA_PATH}/valid.y
export TEST_SOURCES=${DATA_PATH}/test.x
export TEST_TARGETS=${DATA_PATH}/test.y

export TRAIN_STEPS=1000000

export CUDA_VISIBLE_DEVICES=5

export MODEL_DIR=${TMPDIR:-/tmp}/nmt_conv_seq2seq
export MODEL_DIR=./model/textsum_all
mkdir -p $MODEL_DIR

export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 256 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --gpu_allow_growth True



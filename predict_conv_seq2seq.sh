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


export MODEL_DIR=${TMPDIR:-/tmp}/nmt_conv_seq2seq
export MODEL_DIR=./model/textsum_all
mkdir -p $MODEL_DIR

export PRED_DIR=./predict


###with beam search
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt


./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt



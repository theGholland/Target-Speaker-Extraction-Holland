PYTHON ?= python

VOICES_DIR ?= data/voices
OUT_DIR ?= out_eval
NUM_SPEAKERS ?= 10
SUBSET ?= test-clean
SNR_DB ?= 0
NUM_BABBLE ?= 3
SNR_LIST ?= -5,0,5
BABBLE_LIST ?= 1,2,3
SEP_MODELS ?= dprnn

.PHONY: prepare eval sweep_snr sweep_babble grid clean

prepare:
	$(PYTHON) src/prepare_voices.py --out $(VOICES_DIR) --num_speakers $(NUM_SPEAKERS) --subset $(SUBSET)

eval:
	$(PYTHON) src/eval_tse_on_voices.py --voices_dir $(VOICES_DIR) --out_dir $(OUT_DIR) --snr_db $(SNR_DB) --num_babble_voices $(NUM_BABBLE) --sep_models $(SEP_MODELS)

sweep_snr:
	$(PYTHON) src/eval_tse_on_voices.py --voices_dir $(VOICES_DIR) --out_dir $(OUT_DIR) --snr_list "$(SNR_LIST)" --num_babble_voices $(NUM_BABBLE) --sep_models $(SEP_MODELS)

sweep_babble:
	$(PYTHON) src/eval_tse_on_voices.py --voices_dir $(VOICES_DIR) --out_dir $(OUT_DIR) --snr_db $(SNR_DB) --babble_list "$(BABBLE_LIST)" --sep_models $(SEP_MODELS)

grid:
	$(PYTHON) src/eval_tse_on_voices.py --voices_dir $(VOICES_DIR) --out_dir $(OUT_DIR) --snr_list "$(SNR_LIST)" --babble_list "$(BABBLE_LIST)" --sep_models $(SEP_MODELS)

clean:
	rm -rf $(OUT_DIR)

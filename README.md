##


## Requirements
mkdir -p "$HOME/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12

source .venv/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

uv pip install -r requirements.txt

## Tanl extractor usage
python tanl_extractor.py --model_dir tanl-scierc_all-step44270 --tokenizer_dir t5-base --split train --task scierc_joint_er --num_beams 2 --max_input_len 512 --max_output_len 1024 --out scico_train_tanl_extraction.jsonl; python tanl_extractor.py --model_dir tanl-scierc_all-step44270 --tokenizer_dir t5-base --split dev --task scierc_joint_er --num_beams 2 --max_input_len 512 --max_output_len 1024 --out scico_dev_tanl_extraction.jsonl; python tanl_extractor.py --model_dir tanl-scierc_all-step44270 --tokenizer_dir t5-base --split test --task scierc_joint_er --num_beams 2 --max_input_len 512 --max_output_len 1024 --out scico_test_tanl_extraction.jsonl



## Build signature yaml
python build_signature.py --pred_path scico_dev_tanl_extraction.jsonl --split validation --out_path scico_signatures_dev.jsonl

## Cross-Encoder

python train_signature_coref.py --signatures_path_train data_tanl/scico_signatures_train.jsonl --signatures_path_val data_tanl/scico_signatures_dev.jsonl  --epochs 3 --batch_size 8 --max_length 512 --neg_pos_ratio 1.0 --output_dir ckpts_sigce

python predict_signature_coref.py --split test --signatures_path data_tanl/scico_signatures_test.jsonl --checkpoint ckpts_sigce/best_epoch1_f10.8654.pt --distance_threshold 0.5 --out_path predicted_clusters.jsonl

python eval/run_coref_eval.py --split test --predicted_clusters predicted_clusters.jsonl

## Eval

python utils/make_scico_pred_jsonl.py --split test --predicted_clusters output/predicted_clusters.jsonl --out_path output/system_pred.jsonl


python evaluate.py data_scico/test.jsonl output/system_pred.jsonl

## Calibrate
python -m calibration.dump_pair_scores --split validation --signatures_path data_tanl/scico_signatures_dev.jsonl --checkpoint ckpts_sigce/best_epoch1_f10.8654.pt --out_path output/pair_scores_dev.jsonl

python -m calibration.sweep_thresholds --scores_path output/pair_scores_dev.jsonl --split validation --eval_script evaluate_signature_coref.py --method agglomerative --linkage average --t_min 0.10 --t_max 0.90 --t_step 0.1 --temperature 1.0 --work_dir sweep_dev

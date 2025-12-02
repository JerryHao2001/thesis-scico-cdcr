##
Download code with 
git clone https://github.com/JerryHao2001/thesis-scico-cdcr.git

get data and weights to 
/data_scico
/data_tanl
/output
/ckpts_sigce
/tanl-scierc_all-backbone

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
python preprocess/build_signature.py --pred_path data_tanl/scico_train_tanl_extraction.jsonl --split train --out_path data_tanl/scico_signatures_train.jsonl; 
python preprocess/build_signature.py --pred_path data_tanl/scico_dev_tanl_extraction.jsonl --split validation --out_path data_tanl/scico_signatures_dev.jsonl; 
python preprocess/build_signature.py --pred_path data_tanl/scico_test_tanl_extraction.jsonl --split test --out_path data_tanl/scico_signatures_test.jsonl


## Cross-Encoder

python train_signature_coref.py --signatures_path_train data_tanl/scico_signatures_train.jsonl --signatures_path_val data_tanl/scico_signatures_dev.jsonl  --epochs 3 --batch_size 8 --max_length 512 --neg_pos_ratio 10.0 --output_dir ckpts_sigce_neg10

python predict_signature_coref.py --split test --signatures_path data_tanl/scico_signatures_test.jsonl --checkpoint ckpts_sigce/best_epoch1_f10.8654.pt --distance_threshold 0.1 --out_path output/predicted_clusters_0.1.jsonl

python eval/run_coref_eval.py --split test --predicted_clusters output/predicted_clusters.jsonl

## Eval

python utils/make_scico_pred_jsonl.py --split test --predicted_clusters output/predicted_clusters_0.1.jsonl --out_path output/system_pred_0.1.jsonl

python evaluate_signature_coref.py data_scico/test.jsonl output/system_pred_0.1.jsonl

## Calibrate
python -m calibration.dump_pair_scores --split validation --signatures_path data_tanl/scico_signatures_dev.jsonl --checkpoint ckpts_sigce_neg10/best_epoch3_f10.8200.pt --out_path output_10neg/pair_scores_dev.jsonl;
python -m calibration.dump_pair_scores --split test --signatures_path data_tanl/scico_signatures_test.jsonl --checkpoint ckpts_sigce_neg10/best_epoch3_f10.8200.pt --out_path output_10neg/pair_scores_test.jsonl;
python -m calibration.fit_temperature_from_pairs --scores_path output_10neg/pair_scores_dev.jsonl --split validation --out_json output_10neg/temperature_dev.json;
python -m calibration.sweep_thresholds --scores_path output_10neg/pair_scores_dev.jsonl --split validation --eval_module_path evaluate_signature_coref.py --temperature_json output_10neg/temperature_dev.json --method agglomerative --linkage average --t_min 0.1 --t_max 0.9 --t_step 0.05
python -m calibration.sweep_thresholds --scores_path output_10neg/pair_scores_test.jsonl --split test --eval_module_path evaluate_signature_coref.py --temperature_json output_10neg/temperature_dev.json --method agglomerative --linkage average --t_min 0.1 --t_max 0.9 --t_step 0.1

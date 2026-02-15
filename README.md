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
python preprocess/build_signature.py --pred_path data_tanl/scico_dev_tanl_extraction.jsonl --split validation --out_path data_tanl/scico_signatures_dev.jsonl; 
python preprocess/build_signature.py --pred_path data_tanl/scico_train_tanl_extraction.jsonl --split train --out_path data_tanl/scico_signatures_train.jsonl; 
python preprocess/build_signature.py --pred_path data_tanl/scico_test_tanl_extraction.jsonl --split test --out_path data_tanl/scico_signatures_test.jsonl


## Cross-Encoder

python train_signature_coref.py --signatures_path_train data_tanl/scico_signatures_train.jsonl --signatures_path_val data_tanl/scico_signatures_dev.jsonl  --bert_model allenai/specter2_base --adapter_name allenai/specter2  --epochs 5 --batch_size 8 --max_length 512 --neg_pos_ratio 10 --output_dir ckpts/ckpts_sigce_specter1010

python train_signature_antecedent.py --signatures_path_train data_tanl/scico_signatures_train.jsonl --signatures_path_val data_tanl/scico_signatures_dev.jsonl --bert_model allenai/scibert_scivocab_uncased --epochs 3 --lr 2e-5 --topic_batch_size 1 --pair_batch_size 1 --cand_strategy hybrid --cand_window 12 --cand_max_candidates 12 --train_eps --eps_init 0.0 --amp --eval_module_path evaluate_signature_coref.py  --output_dir ckpts/antecedent/trail_1_12_12 --save_every_epoch

python train_signature_antecedent_streaming.py --signatures_path_train data_tanl/scico_signatures_train.jsonl --signatures_path_val data_tanl/scico_signatures_dev.jsonl --bert_model allenai/scibert_scivocab_uncased --epochs 5 --lr 2e-5 --topic_batch_size 1 --pair_batch_size 16 --cand_strategy all --cand_window 32 --cand_max_candidates 0 --max_length=384 --train_eps --eps_init 0.0 --eval_module_path evaluate_signature_coref.py --eval_every_epoch --output_dir ckpts/antecedent/trail_1_16_all_0_384 --save_every_epoch --amp



python predict_signature_coref.py --split test --signatures_path data_tanl/scico_signatures_test.jsonl --checkpoint ckpts_sigce/best_epoch1_f10.8654.pt --distance_threshold 0.1 --out_path output/predicted_clusters_0.1.jsonl

python eval/run_coref_eval.py --split test --predicted_clusters output/predicted_clusters.jsonl

## Eval

python utils/make_scico_pred_jsonl.py --split test --predicted_clusters output/predicted_clusters_0.1.jsonl --out_path output/system_pred_0.1.jsonl

python evaluate_signature_coref.py data_scico/test.jsonl output/system_pred_0.1.jsonl

## Calibrate
python -m calibration.dump_pair_scores --split validation --signatures_path data_tanl/scico_signatures_dev.jsonl --bert_model allenai/specter2_base --adapter_name allenai/specter2 --checkpoint ckpts/ckpts_sigce_specter10/best_epoch3_f10.8455.pt --out_path output/output_specter10/pair_scores_dev.jsonl;
python -m calibration.dump_pair_scores --split test --signatures_path data_tanl/scico_signatures_test.jsonl --bert_model allenai/specter2_base --adapter_name allenai/specter2 --checkpoint ckpts/ckpts_sigce_specter10/best_epoch3_f10.8455.pt --out_path output/output_specter10/pair_scores_test.jsonl;
python -m calibration.fit_temperature_from_pairs --scores_path output/output_specter10/pair_scores_dev.jsonl --split validation --out_json output/output_specter10/temperature_dev.json;
python -m calibration.sweep_thresholds --scores_path output/output_specter10/pair_scores_dev.jsonl --split validation --eval_module_path evaluate_signature_coref.py --temperature_json output/output_specter10/temperature_dev.json --method agglomerative --linkage average --t_min 0.1 --t_max 0.9 --t_step 0.02

python -m calibration.sweep_thresholds --scores_path output/output_specter10/pair_scores_test.jsonl --split test --eval_module_path evaluate_signature_coref.py --temperature_json output/output_specter10/temperature_dev.json --method agglomerative --linkage average --t_min 0.3 --t_max 0.3 --t_step 0.1

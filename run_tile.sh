python run_infer.py \
--gpu='0' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=original \
--model_path=./models/pretrained/hovernet_consep_consep_testing.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=dataset/CoNSeP/Test/Images/ \
--output_dir=dataset/CoNSeP/Pred_pretrained/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath

$SHELL
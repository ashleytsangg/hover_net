python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=dataset/Lymphocyte/type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=./models/pretrained/monusac_lymph_10_fix.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=dataset/Lymphocyte/Test/Images \
--output_dir=out/Lymphocyte/monusac_lymph_10_fix \
--mem_usage=0.1 \
--draw_dot \
--save_qupath

$SHELL
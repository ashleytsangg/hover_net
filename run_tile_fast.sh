date
python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\type_info.json' \
--batch_size=64 \
--model_mode=fast \
--model_path='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\0921 lymph models\full branch\pannuke_10_WLM2_CEL.tar' \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\All\Images' \
--output_dir='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\Lymphocyte\0921 data\full branch\pannuke_10_WLM2_CEL' \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
date
$SHELL
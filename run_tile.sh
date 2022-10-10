date
python run_infer.py \
--gpu='0' \
--nr_types=0 \
--type_info_path= \
--batch_size=64 \
--model_mode=original \
--model_path='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\segmentation\IF\full_seg_10_np_hv_B5_unfilt.tar' \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\IF\Test\Images' \
--output_dir='\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\out\IF\full_seg_10_np_hv_B5_unfilt' \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
date
$SHELL
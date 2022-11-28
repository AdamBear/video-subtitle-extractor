#export CUDA_VISIBLE_DEVICES=0
#nohup hub serving start -c hub_serving/ocr_system/config.json > ocr.log 2>&1 & echo $! > pid_ocr.txt
nohup hub serving start -m ch_pp-ocrv3_det > ocr.log 2>&1 & echo $! > pid_ocr.txt
# chinese_ocr_db_crnn_server
# hub serving start -m ch_pp-ocrv3_det
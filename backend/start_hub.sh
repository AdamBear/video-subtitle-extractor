#export CUDA_VISIBLE_DEVICES=0
#nohup hub serving start -c hub_serving/ocr_system/config.json > ocr.log 2>&1 & echo $! > pid_ocr.txt
nohup hub serving start -m chinese_ocr_db_crnn_server -p 8866 > ocr.log 2>&1 & echo $! > pid_ocr.txt

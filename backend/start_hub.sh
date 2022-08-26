export CUDA_VISIBLE_DEVICES=0
nohup hub serving start -c hub_serving/ocr_system/config.json > ocr.log 2>&1 & echo $! > pid_ocr.txt

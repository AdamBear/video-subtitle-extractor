check_process(){
	PID=`cat pid_ocr.txt`
	PID_EXIST=$(ps aux | awk '{print $2}'| grep -w $PID)

	if [ ! $PID_EXIST ];then
		echo the process $PID is not exist
		return 0
	else
		echo the process $PID exist
		return 1
	fi
}


# Check whether the instance of thread exists:
while [ 1 ] ; do
        echo 'begin checking...'
        check_process
        if [ $? -eq 0 ]; # none exist
        then
		echo restart
                sh start_hub.sh
        fi
        sleep 60
done

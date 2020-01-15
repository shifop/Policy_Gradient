#! /bin/bash
echo "current PID: $$"
nohup python main.py  > log.log 2>&1 &
echo "$!"
echo "$!" > pid
#!/bin/bash

TERM=xterm
SHELL=/bin/bash
XDG_SESSION_COOKIE=8342d40a26d074e931c6e4e600000004-1393211401.179579-245733898
USER=dosenet/radwatch-airmonitor
MAIL=/var/mail/dosenet/radwatch-airmonitor
#PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
PATH=/home/dosenet/anaconda3/bin:/home/dosenet/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
PWD=/home/dosenet/radwatch-airmonitor
LANG=en_US.UTF-8
SHLVL=1
HOME=/home/dosenet/radwatch-airmonitor
LOGNAME=dosenet/radwatch-airmonitor
DISPLAY=localhost:10.0
#_=/usr/bin/env

date;

cd /home/dosenet/radwatch-airmonitor/image_scripts/analysis/
pwd > /home/dosenet/radwatch-airmonitor/image_scripts/analysis/out.txt
python3 /home/dosenet/radwatch-airmonitor/image_scripts/weather_gatherer.py
python3 /home/dosenet/radwatch-airmonitor/image_scripts/analysis/raw_analysis.py
python3 /home/dosenet/radwatch-airmonitor/image_scripts/analysis/h5_analysis.py
#python /home/dosenet/radwatch-airmonitor/image_scripts/analysis/stage_h5.py
convert -geometry 300x220+0+0 iso_One_Day.png iso_One_Day_small.png
mkdir -p "rooftop_tmp"
mv *.png ./rooftop_tmp
mv weather_sorted.csv ./rooftop_tmp
#mv weather_bq.csv ./rooftop_tmp
#tar cvf rooftop.tar rooftop_tmp
#scp rooftop.tar rpavlovs@kepler.berkeley.edu:/tmp
#ssh rpavlovs@kepler.berkeley.edu 'bash -s' < unpacking_script.sh
env >> /home/dosenet/radwatch-airmonitor/image_scripts/analysis/out.txt
#kill $SSH_AGENT_PID
lftp -e "set sftp:auto-confirm yes; mirror -Rnv /home/dosenet/radwatch-airmonitor/image_scripts/analysis/rooftop_tmp /test/; quit;" -u coeradwatch-RADWATCH,'x9DvsvP9gbVWT9F' sftp://coeradwatch.sftp.wpengine.com:2222
#!/bin/bash

TERM=xterm
SHELL=/bin/bash
XDG_SESSION_COOKIE=8342d40a26d074e931c6e4e600000004-1393211401.179579-245733898
USER=rtpavlovsk21
MAIL=/var/mail/rtpavlovsk21
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
PWD=/home/rtpavlovsk21
LANG=en_US.UTF-8
SHLVL=1
HOME=/home/rtpavlovsk21
LOGNAME=rtpavlovsk21
DISPLAY=localhost:10.0
_=/usr/bin/env

date;
AGENT_INFO=/home/rtpavlovsk21/.agent_info; export AGENT_INFO
eval `ssh-agent -s`
ssh-add ~/.fake/kepler_unpro

cd /home/rtpavlovsk21/image_scripts/analysis/
pwd > /home/rtpavlovsk21/image_scripts/analysis/out.txt
python /home/rtpavlovsk21/image_scripts/analysis/raw_analysis.py
python /home/rtpavlovsk21/image_scripts/analysis/h5_analysis.py
#python /home/rtpavlovsk21/image_scripts/analysis/stage_h5.py
convert -geometry 300x220+0+0 iso_One_Day.png iso_One_Day_small.png
mkdir rooftop_tmp
mv *.png ./rooftop_tmp
mv weather.csv ./rooftop_tmp
mv weather_bq.csv ./rooftop_tmp
tar cvf rooftop.tar rooftop_tmp
scp rooftop.tar rpavlovs@kepler.berkeley.edu:/tmp
ssh rpavlovs@kepler.berkeley.edu 'bash -s' < unpacking_script.sh
env >> /home/rtpavlovsk21/image_scripts/analysis/out.txt
kill $SSH_AGENT_PID
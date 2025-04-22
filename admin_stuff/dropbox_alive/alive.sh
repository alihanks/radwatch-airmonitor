#!/bin/bash
export db="python $HOME/radwatch-airmonitor/.dropbox/dropbox.py"
export k=`pgrep dropbox`

if [ -z "$k" ]; then
    echo "not running!"
    $db status
    $db start -i
else
    echo $k
fi

#To-Do: Hack to keep video files from filling the drive
# rm Dropbox/Apps/CRaNBARI\ \(1\)/breezeway/video/*

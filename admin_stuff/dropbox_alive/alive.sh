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

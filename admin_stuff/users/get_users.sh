#find /home -maxdepth 1 -type d  > users.txt
#sed -i 1,2d users.txt
#cat users.txt | xargs stat -c "%U" > users.txt.whatever
#cat users.txt.whatever | xargs sudo chage -W 30 -M 365

for file in $( find /home -maxdepth 1 -type d ); do 
    tmp=`stat -c "%U" $file`
    if [ $tmp != 'root' ]; then
       echo "-----"
       echo $tmp
       sudo chage -l $tmp
    fi
done
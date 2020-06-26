for i in `ls /var/mail | grep -v root`;
do
    len=`wc -l /var/mail/$i | cut -f1 -d' '`;
    echo $len;
    if [ "$len" -gt "1000" ]; then
	tail -100 /var/mail/$i >> /var/mail/tmp;
	mv /var/mail/tmp /var/mail/$i;
    fi
    sudo chown $i /var/mail/$i;
done
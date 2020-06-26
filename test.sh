#!/bin/bash
sudo cat /var/log/auth.log | grep -E "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | grep "Accepted" | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" > ips.dat

while read line
do
    name=$line
    whois $name | grep OrgName >> ip_proc.dat
done < ips.dat

awk '!x[$0]++' ip_proc.dat

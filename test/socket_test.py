import socket
import time;
import datetime;
import numpy as np;

udp_ip="192.168.1.101";
udp_port=5005;
message='';
time.sleep(2);
sock=[];
for x in range(0,3):
    sock.append(socket.socket(socket.AF_INET,socket.SOCK_DGRAM));

while True:
    for x in range(0,len(sock)):
        message=str(x)+","+str(datetime.datetime.now())+","+str(np.random.poisson(100));
        sock[x].sendto(message,(udp_ip,udp_port));
        time.sleep(1);

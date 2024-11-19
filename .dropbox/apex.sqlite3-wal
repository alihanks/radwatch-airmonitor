import socket;

udp_ip="192.168.1.101";
udp_port=5005;

sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);
sock.bind((udp_ip,udp_port));

while True:
    data,addr=sock.recvfrom(2048);
    print data;

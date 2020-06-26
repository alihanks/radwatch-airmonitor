from suds.client import Client

url='https://localhost:7080/webservices/WebServiceTestBean?wsdl';
client=Client(url);

print client;

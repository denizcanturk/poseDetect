# create a UDP Manager class that will handle the UDP communication
# it will send the data to the server and receive the data from the server
# it will also handle the connection and disconnection of the server
# it will also handle the error handling
# it will also handle the logging
# it will also handle the configuration
# it will also handle the statistics

import socket

class UDPManager:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self, data:str):
        #convert data to bytes
        data = data.encode("utf-8") 
        
        self.socket.sendto(data, (self.server_ip, self.server_port))

    def receive_data(self):
        return self.socket.recvfrom(1024)
    
    def close(self):
        self.socket.close()

if __name__ == "__main__":
    # Example usage:
    udp_manager = UDPManager("127.0.0.1", 12345)
    udp_manager.send_data("Hello, server!")
    print(udp_manager.receive_data())
    udp_manager.close()

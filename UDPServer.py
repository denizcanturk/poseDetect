#Create a UDP server class that will receive the data from the client
# and will process the data
# and will send the data to the client
# it will also handle the connection and disconnection of the client
# it will also handle the error handling
# it will also handle the logging
# it will also handle the configuration
# it will also handle the statistics

import socket

class UDPServer:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        self.socket.bind((self.server_ip, self.server_port))
        print(f"Server started on {self.server_ip}:{self.server_port}")

    def receive_data(self):
        return self.socket.recvfrom(1024)
    
    def close(self):
        self.socket.close()

if __name__ == "__main__":
    server = UDPServer("127.0.0.1", 12345)
    server.start()
    
    try:
        while True:
            data, addr = server.receive_data()
            print(data)
    except KeyboardInterrupt:
        print("Closing the server...")
    finally:
        server.close()

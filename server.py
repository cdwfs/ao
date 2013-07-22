import os
import sys

port = 80

if sys.version_info.major == 3:
    import http.server
    import socketserver
    server_base_class = socketserver.TCPServer
    class LocalTCPServer(socketserver.TCPServer):
        "Only accepts requests from 127.0.0.1"
        def verify_request(self, request, client_address):
            return (client_address[0] == '127.0.0.1')
    handler = http.server.SimpleHTTPRequestHandler
elif sys.version_info.major == 2:
    import SimpleHTTPServer
    import SocketServer
    class LocalTCPServer(SocketServer.TCPServer):
        "Only accepts requests from 127.0.0.1"
        def verify_request(self, request, client_address):
            return (client_address[0] == '127.0.0.1')
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = LocalTCPServer(("",port), handler)
if port == 80:
    server_string = "localhost"
else:
    server_string = "localhost:%d" % port
print("Serving %s at http://%s\n" % (os.getcwd(),server_string))
httpd.serve_forever()

import os
import sys

port = 80

if sys.version_info.major == 3:
    import http.server
    import socketserver
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("",port), handler)
elif sys.version_info.major == 2:
    import SimpleHTTPServer
    import SocketServer
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("",port), handler)
if port == 80:
    server_string = "localhost"
else:
    server_string = "localhost:%d" % port
print("Serving %s at http://%s\n" % (os.getcwd(),server_string))
httpd.serve_forever()

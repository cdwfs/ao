import sys

port = 8000

if sys.version_info.major == 3:
    import http.server
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("",port), handler)
elif sys.version_info.major == 2:
    import SimpleHTTPServer
    import SocketServer
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("",port), handler)

print("Serving at port %d\n" % port)
httpd.serve_forever()

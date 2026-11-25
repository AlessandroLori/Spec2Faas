"""Proxy CONNECT isolato per una sola fixture HTTPS attestata.

Eseguito esclusivamente nel container di test, non importato dalla pipeline.
"""
import socketserver
import ssl
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path('/opt/fixture')
URL = urlsplit((ROOT / 'url.txt').read_text())
DATA = (ROOT / 'fixture.bin').read_bytes()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(8)
        try:
            line = self.rfile.readline(8192).decode('latin1').strip()
            fields = line.split(' ')
            if len(fields) != 3 or fields[0] != 'CONNECT' or fields[1] != URL.hostname + ':443':
                self.wfile.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                return
            for _ in range(60):
                header = self.rfile.readline(8192)
                if not header or header in (b'\r\n', b'\n'):
                    break
            self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
            self.wfile.flush()
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(str(ROOT / 'cert.pem'), str(ROOT / 'key.pem'))
            sock = tls.wrap_socket(self.connection, server_side=True)
            sock.settimeout(8)
            try:
                stream = sock.makefile('rb')
                request_line = stream.readline(8192).decode('latin1').strip().split(' ')
                expected_path = URL.path + (('?' + URL.query) if URL.query else '')
                if len(request_line) != 3 or request_line[0] != 'GET' or request_line[1] != expected_path:
                    sock.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                    return
                for _ in range(60):
                    header = stream.readline(8192)
                    if not header or header in (b'\r\n', b'\n'):
                        break
                sock.sendall((f'HTTP/1.1 200 OK\r\nContent-Length: {len(DATA)}\r\n'
                              'Content-Type: application/octet-stream\r\n'
                              'Connection: close\r\n\r\n').encode('ascii') + DATA)
            finally:
                sock.close()
        except (OSError, ssl.SSLError, UnicodeError, ValueError):
            return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    with Server(('0.0.0.0', 8181), Handler) as server:
        server.serve_forever(poll_interval=.2)

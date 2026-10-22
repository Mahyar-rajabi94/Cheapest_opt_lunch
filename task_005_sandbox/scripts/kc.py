"""Client: send a file's (or stdin's) code to the kernel and print its output."""
import socket, struct, sys
SOCK = "\0" + sys.argv[1]
code = open(sys.argv[2]).read() if len(sys.argv) > 2 else sys.stdin.read()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(SOCK)
data = code.encode()
s.sendall(struct.pack("!Q", len(data)) + data)
def recv_all(n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf
n = struct.unpack("!Q", recv_all(8))[0]
sys.stdout.write(recv_all(n).decode())

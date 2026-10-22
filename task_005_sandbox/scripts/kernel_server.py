"""Minimal persistent exec kernel over a Unix socket (single process => single harness instance)."""
import contextlib, io, os, socket, struct, sys, threading, traceback

SOCK = "\0" + sys.argv[1]
REPO = sys.argv[2]
os.chdir(REPO)
sys.path.insert(0, REPO)
ns = {"__name__": "__kernel__"}
lock = threading.Lock()

def recv_all(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf

def handle(conn):
    with conn:
        n = struct.unpack("!Q", recv_all(conn, 8))[0]
        code = recv_all(conn, n).decode()
        out = io.StringIO()
        with lock, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                exec(compile(code, "<kernel>", "exec"), ns)
            except BaseException:
                traceback.print_exc()
        data = out.getvalue().encode()
        conn.sendall(struct.pack("!Q", len(data)) + data)

if False:
    os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(4)
print("kernel ready", flush=True)
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()

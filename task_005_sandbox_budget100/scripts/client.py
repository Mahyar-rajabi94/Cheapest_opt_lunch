import os
import sys
from multiprocessing.connection import Client

os.chdir(os.path.dirname(os.path.abspath(__file__)))
code = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
c = Client("srv.sock", family="AF_UNIX", authkey=b"task005")
c.send(code)
print(c.recv())
c.close()

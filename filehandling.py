import os
try:
    with open("README.md","a") as wr:
        wr.write("EnXOR1212")
except:
    FileNotFoundError
with open("README.md","r") as logfile:
    content=logfile.readlines()
    print(content)
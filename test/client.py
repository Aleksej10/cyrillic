import socket, sys



if __name__ == "__main__":
    host = socket.gethostname()
    port = 8081
    client = socket.socket()
    try: 
        client.connect((host,port))
    except ConnectionRefusedError:
        print('make sure cyr.daemon is running')
        sys.exit()

    msg = input()
    client.send(msg.encode())
    print(client.recv(2048).decode())

    client.close()

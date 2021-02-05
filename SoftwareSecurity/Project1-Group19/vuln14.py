from twisted.web import proxy, http
from twisted.internet import reactor
from twisted.python import log
import sys
import os

os.system('sleep 2 && curl -X POST --data \'username=pentest1&password=pentest1\' http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt/login --proxy 127.0.0.1:8899 &')

log.startLogging(sys.stdout)

class MyProxy(proxy.Proxy):
    def dataReceived(self, data):

        d = data.decode('utf-8')
        print(d)
        assert 'username=pentest1&password=pentest1' in d

        return proxy.Proxy.dataReceived(self, data)

class ProxyFactory(http.HTTPFactory):
  protocol=MyProxy

def stop_pls():
    reactor.stop()

factory = ProxyFactory()
reactor.listenTCP(8899, factory)
reactor.callLater(4, stop_pls)
reactor.run()

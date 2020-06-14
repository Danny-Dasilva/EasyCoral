import subprocess
import csv
import os
from .wifi import Cell, Scheme
from subprocess import check_output
import re

def list_connected():
    scanoutput = check_output(["nmcli", "-t", "-f", "name", "connection", "show", "--active"])

    return [ line.decode() for line in scanoutput.split()]

def search_wifi():
    wifi_list = {}
    cells = Cell.all('wlan0')
    for connection in cells:
        SSID = connection.ssid
        enc = connection.encrypted
        if SSID:
            wifi_list[SSID] = enc
    connected = list_connected()
    for network in connected:
        if network in wifi_list:
            wifi_list[network] = 'connected'
    return wifi_list




def write_previous_connections(SSID):
    ssids = read_json('previous')
    if SSID not in ssids:

        ssids.append(SSID)
    
    write_json('previous', ssids)


def search_connected():
    process = subprocess.Popen(['nmcli', 'd'], stdout=subprocess.PIPE)
    #process = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE)

    stdout, stderr = process.communicate()
    #print(stdout.decode('utf-8').splitlines())
    reader = csv.DictReader(stdout.decode('utf-8').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['NAME', 'UUID',
                                        'TYPE', 'DEVICE',])

    wifi_list = {}
    connected = {'connection' : False}
    for row in reader:
        if row ['TYPE'] == 'wifi':
            uuid = row['UUID']
            name = row['NAME']
            wifi_list[name] = connection
    return connected, wifi_list

def del_previous(name):
    process = subprocess.Popen(['nmcli', 'connection'], stdout=subprocess.PIPE)
    #process = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE)

    stdout, stderr = process.communicate()
    # print(stdout.decode('utf-8').splitlines())
    r = stdout.decode('utf-8').splitlines()
    pattern = "(.*?)[ ]{2,}(.*?)[ ]{2,}(.*?)[ ]{2,}(.*?)[ ]{2,}(.*?)[ ]{2,}(.*?)[ ]{2,}(.*?)"
    
    for lines in r[1:]:
        m = re.search(pattern, lines)
        if m:
            ssid = m.group(1)
            key = m.group(2)
            if name in ssid:
                os.system(f"nmcli con delete {key}")
    return True

#  nmcli d up <name>
def disconnect(wifi_name):
    os.system(f'nmcli con down id {wifi_name}')

def request_password():
    # send a socket requesting a password
    return False

def connect_wifi(data):
    for ssid, password in data.items(): 
      
        F = Finder(server_name=ssid,
                    password=password)
        return F.run()

 



class Finder:
    def __init__(self, *args, **kwargs):
        self.server_name = kwargs['server_name']
        self.password = kwargs['password']
        self.interface_name = 'wlan0'
        self.main_dict = {}

    def run(self):
        del_previous(self.server_name)
        try:
            result = self.connection(self.server_name)
        except Exception as exp:
            return f"Error couldn't connect to: {self.server_name}. {exp}"
        else:
            if result:
               return f"Successfully connected to {self.server_name}"

    def connection(self, name):
        try:
            # t = os.popen(f"nmcli d wifi connect {name} password {self.password} ").read()
            output = subprocess.check_output(f"nmcli d wifi connect {name} password {self.password} ",  shell=True)

            result = output.decode()
            if "Error" in result:
                raise ValueError("Password is Incorrect")

        except:
            raise
        else:
            return True

if __name__ == "__main__":
    # Server_name is a case insensitive string, and/or regex pattern which demonstrates
    # the name of targeted WIFI device or a unique part of it.
    r = connect_wifi({'Dasilva-Wifi' : 'orangejet'})
    print(r)
    # t = search_wifi()
    # print(t)
    # del_previous('Dasilva-Wifi')

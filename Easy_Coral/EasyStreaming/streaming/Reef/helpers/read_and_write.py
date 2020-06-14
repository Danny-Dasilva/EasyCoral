
import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
def read_json(filename):
    with open(f'{dir_path}/../assets/json/{filename}.json', 'r') as json_file:

        data = json.load(json_file)
        
    return data

def write_json_2(filename, data):
    
    with open(f'{dir_path}/../assets/json/{filename}.json', 'w') as outfile:
        json.dump(data, outfile)
def append_json(filename, data):
    with open(f'{dir_path}/../assets/json/{filename}.json', 'w') as outfile:
        json.dump(data, outfile)
def json_parse(data):
    data = json.dumps(data)
    data = json.loads(data)
    return(data)


def write_json(filename, nput):
    with open(f'{dir_path}/../assets/json/{filename}.json') as f:

        try:
            data = json.load(f)
            data.update(nput)      
        except:
            data = nput
       


    with open(f'{dir_path}/../assets/json/{filename}.json', 'w') as f:
        json.dump(data, f)

def delete_json(filename, nput):
    with open(f'{dir_path}/../assets/json/{filename}.json') as f:
        
        
        data = json.load(f)
        print(data, "input", nput)
        del data[nput] 

    with open(f'{dir_path}/../assets/json/{filename}.json', 'w') as f:
        json.dump(data, f)
    f.close()


if __name__ == '__main__':
    json_data = {}
    write_json('profiles', json_data)
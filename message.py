import json
import serial

'''
ser = serial.Serial(
        port='COM3',
        baudrate=57600
    )
    '''

def receive(ser):
    
    '''
    # receive
    ser.flush()
    res = ser.readline()
    data = res.decode()[:len(res)-1]
    # ser.flush()

    dict_json = json.loads(data, strict=False)

    print(dict_json)

    return dict_json
    '''

    
    data = ""
    while True:
        char = ser.read().decode('utf-8')
        if char == '':  # No more data to read
            return 0

        data += char
        if char == '\n':  # End of the JSON object
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                return 0
        # ser.flush()
        # print(char)
    
    

def send(ser, steer):
    doc = {"steer":steer}
    blank = '\n'
    ser.write(json.dumps(doc).encode('utf-8'))
    ser.write(blank.encode('utf-8'))
    ser.flush()

'''
while(1):
    send(ser, 1)
    print(1)
    send(ser, 3)
    print(3)
    send(ser, 5)
    print(5)
    dict_json = receive(ser)
    print(dict_json)
    '''



        

#!/usr/bin/env python3
"""Read gas meter and publish as MQTT"""
import sys
import os
import json
from datetime import datetime
from datetime import timedelta
import time
import statistics
import collections
import paho.mqtt.client as mqttClient
import cv2
import gas_meter_reader

CONNECTED = False # MQTT connected
RANGEFILE = 'expected_range.json'


def on_connect(client, userdata, flags, code):
    """Connect completion for Paho"""
    _ = client
    _ = userdata
    _ = flags
    global CONNECTED
    if code == 0:
        print("Connected to broker")
        CONNECTED = True                #Signal connection
    else:
        print("Connection failed")

def get_average_circles(in_vals, mean=False):
    """Compute median or mean of each column of circles in a list
    of columns of circles"""
    val_size = len(in_vals[0])
    columns = []
    for circlelist in range(val_size):
        columns.append([])
    for circlelist in in_vals:
        for column in range(val_size):
            columns[column].append(circlelist[column])
    result = []
    for column in columns:
        x_coords = []
        y_coords = []
        radii = []
        for circle in column:
            x_coords.append(circle[0])
            y_coords.append(circle[1])
            radii.append(circle[2])
        if mean:
            x_average = statistics.mean(x_coords)
            y_average = statistics.mean(y_coords)
            radius_average = statistics.mean(radii)
        else:
            x_average = statistics.median(x_coords)
            y_average = statistics.median(y_coords)
            radius_average = statistics.median(radii)
        result.append([x_average, y_average, radius_average])

    return result



def adjust_range(expected_range, reading):
    print ( "Range: " + str(expected_range[1]) + " ; " + str(expected_range[0]) )
    print ('Reading: ' + str(reading))
    """Adjust reading to be in range, and compute new range if needed"""
    """
    Approach: 
    - Take current reading and ensure it is less than 100 from current reading.   
    -   Previous Reading = Current Reading.
    - Else
    -  Are the last 3 digits less than 100.   
    -  Use the 3 digits and ignore the first digit.
    """
    if expected_range:
        delta = expected_range[1] - expected_range[0]
        print ( "Delta: " + str(delta) )
        while reading > expected_range[1]:
            reading = reading - delta
            print("Adjust downward to %s" % str(reading))
        while reading < expected_range[0]:
            reading = reading + delta
            print("Adjust upward to %s" % str(reading))
    mid = round(reading/500.0) * 500.0
    print ("Mid: " + str(mid))
    new_range = [mid - 1000, mid + 1000]
    return (new_range, reading)

def publish_result(client, reading, last_reading, now):
    """Write result to MQTT or save debug output"""
    # Round to nearest 0.2
    rounded = round(reading * 5.0) / 5.0
    print ("Rounded: "+  str(rounded))
    print("Last Reading: " + str(last_reading))
    
    if last_reading and abs(last_reading - rounded) > 1.0:
        print("Bad Reading: " + str((last_reading and abs(last_reading - rounded) > 1.0)))
        bad_reading = str(round(float(rounded), 1))
        print("Rejecting bad reading %s" % bad_reading)
        debdir = 'output-%s' % bad_reading
        if os.path.isdir('output') and not os.path.isdir(debdir):
            os.rename('output', debdir)
            os.mkdir('output')
        rounded = last_reading
    if last_reading and rounded < last_reading:
        print("Reusing last higher reading %s > %s" %
              (str(last_reading), str(rounded)))
        rounded = last_reading
    message = {"reading": rounded,
               "timestamp": str(now)}
    print("Publish %s" % json.dumps(message))
    
    return rounded

def connect_mqtt():
    """Connect to MQTT"""
    global CONNECTED

    broker_address = "localhost"
    port = 1883
    timeId =  datetime.now().strftime("%H:%M:%S")
    clientId ="dev_gas_meter_" + timeId
    print(clientId)
    client = mqttClient.Client("GasMeter", clientId)    #create new instance
    client.on_connect = on_connect   
    client.username_pw_set("jddayley", "java")         #attach function to callback
    client.connect("192.168.0.116", port=1883) #connect to broker

    client.loop_start()

    while not CONNECTED:
        time.sleep(0.1)
    return client

def get_frames(num_frames):
    """Get the number of video frames specified"""
    frames = []

    cap = cv2.VideoCapture("rtsp://jddayley:java@192.168.0.227/live", cv2.CAP_FFMPEG)
    print ("Connected to Wyze v3") 
    #cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 1024)
    for reading in range(num_frames):
        _ = reading
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_circles(frames):
    """Get circles from the frames"""
    circles_list = []
    images_list = []
    for sample, frame in enumerate(frames):
        try:
            img, circles = gas_meter_reader.get_circles(frame, sample)
            if circles is None:
                continue
            sorted_circles = sorted(circles, key=lambda circle: circle[0])
            #print("Circles: %s" % str(sorted_circles))
            if len(sorted_circles) == 4:
                circles_list.append(sorted_circles)
                images_list.append(img)
        except IndexError:
            print ("Error - Bad Index")
            circles = None
    if not circles_list:
        print("Could not get any circles!")
        circles = None
    else:
        circles = get_average_circles(circles_list)
    return (circles, images_list)

def main(argv):
    """Entry point,  Allow to define an inital reading"""
    _ = argv
    if len(sys.argv) > 1:
        print("reading args")
        last_reading = float(sys.argv[1])
    else:
        last_reading = None
    print (argv)
    client = connect_mqtt()
    err_count = 0
    #expected_range = read_rangefile(RANGEFILE)

    
    # at 5 minute readings, that's an hour
    circle_history = collections.deque(maxlen=12)
    while True:
        now = datetime.now()
        next_time = now + timedelta(minutes=5)
        print("Last Reading: " + str(last_reading))
        frames = get_frames(10)
        print("Got %d frames" % len(frames))

        gas_meter_reader.clear_debug()
        if frames:
            circles, images_list = get_circles(frames)
            if not circles:
                continue
            #print("Median circles: %s" % str(circles))
            circle_history.append(circles)
            circles = get_average_circles(circle_history, mean=True)
            print("Mean history circles: %s" % str(circles))
            readings = []
            for sample, image in enumerate(images_list):
                reading = gas_meter_reader.process(image, circles, sample)
                if len(reading) == 5:
                    print("Reading: %s" % str(reading))
                    output = gas_meter_reader.assemble_reading(reading)
                    readings.append(output)
            reading = statistics.mean(readings)
            print("Mean reading: %s" % str(reading))
            rounded = round(reading * 5.0) / 5.0
            
            # new_expected_range, reading = adjust_range(expected_range, reading)
            # if new_expected_range != expected_range:
            #     expected_range = new_expected_range
            #     with open(RANGEFILE, 'w') as rangefile:
            #         json.dump(expected_range, rangefile)
            if last_reading is None:
                print("First Time")
                last_reading = rounded
            elif rounded < last_reading:
                print("*BAD READING* Reading too low")
                last_reading, rounded = error_check(last_reading, rounded)
            else:
                print("Good - Reading is increasing %s > %s" %
                            (str(last_reading), str(rounded)))
                last_reading, rounded = error_check(last_reading, rounded)
            print ("Rounded: "+  str(rounded))
            print("Last Reading: " + str(last_reading))
            message = {"reading": rounded,
                        "timestamp": str(now)}
            client.publish("gasmeter/reading", json.dumps(message))
            print("Published: " + str(last_reading))
            #last_reading = publish_result(client, reading, last_reading, now)
        else:
            print("Unable to read frames!")

        while datetime.now() < next_time:
            time.sleep(max(timedelta(seconds=0.1),
                           (next_time - datetime.now())/2).total_seconds())

def error_check(last_reading, rounded):
    alterreading = str(last_reading)[:1] + str(rounded)[1:]
    print("It may alter the first digit: " + alterreading)
    if (rounded > last_reading) and ((rounded -last_reading) < 100):
        print("Good One. %s > %s" %
                    (str(last_reading), str(rounded)))
        last_reading = rounded
    elif  ((float(alterreading) > last_reading) and (float(alterreading) - last_reading) < 100 ):
        rounded = float(alterreading)
        last_reading = rounded
        print("Recovered  - Ignore First Digit. Take the last digits %s > %s" %
                    (str(last_reading), str(rounded)))
    else:
        print("BAD READING - no recover. Reusing last higher reading %s > %s" %
                        (str(last_reading), str(rounded)))
        rounded = last_reading

    return last_reading,rounded

if __name__ == '__main__':
    main(sys.argv[1:])
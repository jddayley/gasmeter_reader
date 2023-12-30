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
import mariadb
import sys
from datetime import datetime
from image_predict import *
from globals import *
CONNECTED = False # MQTT connected

path = "output/"

def getDBcon():
    try:
        conn = mariadb.connect(
            user=user,
            password=password,
            host=hostIP,
            port=port,
            database="homeassistant"
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    return conn
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

def connect_mqtt():
    """Connect to MQTT"""
    global CONNECTED

    broker_address = "localhost"
    port = 1883
    timeId =  datetime.now().strftime("%H:%M:%S")
    clientId = mqclientId + timeId
    print(clientId)
    client = mqttClient.Client("GasMeter", clientId)    #create new instance
    client.on_connect = on_connect   
    client.username_pw_set("jddayley", "java")         #attach function to callback
    client.connect(hostIP, port=1883) #connect to broker

    client.loop_start()

    while not CONNECTED:
       time.sleep(sleep_seconds)
    return client

def get_frames(num_frames):
    """Get the number of video frames specified"""
    frames = []
    cap = cv2.VideoCapture(RTSP_HOST, cv2.CAP_FFMPEG)
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
            #print(circles)
            if circles is None:
                print ("No Circles Found")
                continue
            sorted_circles = sorted(circles, key=lambda circle: circle[0])
           
            #circles_list.append(sorted_circles)
            #images_list.append(img)
            if len(sorted_circles) == 4:
                circles_list.append(sorted_circles)
                images_list.append(img)
        except IndexError as err:
           # print(f"Unexpected {err=}, {type(err)=}")
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
        conn = getDBcon()
        cur = conn.cursor()
        cur.execute(
            "SELECT MAX(state) from homeassistant.states where metadata_id='41' AND `state` != 'unknown' AND `state` != 'unavailable'"
            )
        try:
            val = float(cur.fetchone()[0])
            print(val)
        except TypeError:
            #defaul value if new db.
            val = float(8462)
        last_reading = val 

    print (argv)
    client = connect_mqtt()
    err_count = 0
    #expected_range = read_rangefile(RANGEFILE)

    
    # at 5 minute readings, that's an hour
    circle_history = collections.deque(maxlen=12)
    #while True:
    now = datetime.now()
    next_time = now + timedelta(minutes=sleep_time)
    print("Last Reading: " + str(last_reading))
    frames = get_frames(10)
    print("Got %d frames" % len(frames))

    #gas_meter_reader.clear_debug()
    if frames:
        circles, images_list = get_circles(frames)
        #if not circles:
           # continue
        #print("Median circles: %s" % str(circles))
        circle_history.append(circles)
        print(circle_history)
        circles = get_average_circles(circle_history, mean=True)
        print("Mean history circles: %s" % str(circles))
        readings = []
        for sample, image in enumerate(images_list):
            try:
                reading = gas_meter_reader.process(image, circles, sample)
            except TypeError as e:
                print("TypeError Exception:  {e!r}")
                readings.append(1111) 
                break
            except AttributeError:
                print("AttributeError Exception") 
                readings.append(1111) 
                break
            except Exception as e:
                print(f"No idea what happened: {e!r}")
                readings.append(1111) 
                break
            if len(reading) == 5:
                #print("Reading: %s" % str(reading))
                output = gas_meter_reader.assemble_reading(reading)
                readings.append(output)
                
        reading = statistics.mean(readings)
        print("Mean reading: %s" % str(reading))
        rounded = round(reading * 5.0) / 5.0
        dial1 = classify(camera_path + "0-crop-0.jpg")
        dial2 = classify(camera_path + "0-crop-1.jpg")
        dial3 = classify(camera_path + "0-crop-2.jpg")
        dial4 = classify(camera_path + "0-crop-3.jpg")
        ML_predict = int(dial1 + dial2 + dial3 + dial4)
        print ("ML Predict: " + str(ML_predict))
        if last_reading is None:
            print("First Time")
            last_reading = rounded
        elif rounded < last_reading:
            print("*Error Check* reading is the same or too low.")
            if (rounded > last_reading) and ((rounded -last_reading) < 10):
                last_reading, rounded = error_check(last_reading, ML_predict)
            else:
                last_reading, rounded = error_check(last_reading, ML_predict)
            #last_reading, rounded = error_check(last_reading, rounded)
        else:
            print("Checking: value %s rounded %s" %
                        (str(last_reading), str(rounded)))
            if (rounded > last_reading) and ((rounded -last_reading) < 10):
                last_reading, rounded = error_check(last_reading, ML_predict)
            else:
                last_reading, rounded = error_check(last_reading, ML_predict)
        print ("Rounded: "+  str(rounded))
        print("Last Reading: " + str(last_reading))
        message = {"reading": rounded,
                    "timestamp": str(now),
                    "raw": reading, 
                    "ML": ML_predict
                   }
        if publish:
            client.publish(mqqt_q, json.dumps(message))
            print("Published: " + str(last_reading))
            dt_string = now.strftime("%Y/%d/%m %H:%M:%S")
            print("date and time =", dt_string)

    else:
        print("Unable to read frames!")

    # while datetime.now() < next_time:
    #     time.sleep(max(timedelta(seconds=0.1),
    #                    (next_time - datetime.now())/2).total_seconds())

def error_check(last_reading, rounded):
    
    alter1digreading = str(last_reading)[:1] + str(rounded)[1:]
    alter2digreading = str(last_reading)[:2] + str(rounded)[2:]
    alter3digreading = str(last_reading)[:3] + str(rounded)[3:]
   #print("Remove first digit and check: " + alter1digreading)
   #print("Remove second digit and check: " + alter2digreading)
    now = datetime.now()
    dt_string = now.strftime("%Y/%d/%m %H:%M:%S")
    if (int(rounded) == int(last_reading)):
        print("Reading is the Same. %s > %s" %
                            (str(last_reading), str(rounded)))
        last_reading = rounded
    else:
        if (rounded > last_reading) and ((rounded -last_reading) < error_diff):
            print("Good One. %s > %s" %
                        (str(last_reading), str(rounded)))
            last_reading = rounded
        elif  ((float(alter1digreading) > last_reading) and (float(alter1digreading) - last_reading) < error_diff ):
            rounded = float(alter1digreading)
            last_reading = rounded
            os.mkdir(camera_path + "bad/") 
            os.rename(camera_path + "0-crop-0.jpg", camera_path + "bad/0-crop-0.jpg" + dt_string)
            print("Recovered  - Ignore First Digit. Take the last digits %s > %s" %
                        (str(last_reading), str(rounded)))
        elif  ((float(alter2digreading) > last_reading) and (float(alter2digreading) - last_reading) < error_diff ):
            rounded = float(alter2digreading)
            last_reading = rounded
            os.rename(camera_path + "0-crop-1.jpg", camera_path + "bad/0-crop-1.jpg"+ dt_string)
            print("Recovered  - Ignore First and Second Digit. Take the last Two digits %s > %s" %
                        (str(last_reading), str(rounded)))
        elif  ((float(alter3digreading) > last_reading) and (float(alter3digreading) - last_reading) < error_diff ):
            rounded = float(alter3digreading)
            last_reading = rounded
            os.rename(camera_path + "0-crop-2.jpg", camera_path + "bad/0-crop-2.jpg"+ dt_string)
            print("Recovered  - Ignore First and Second Digit. Take the last Two digits %s > %s" %
                        (str(last_reading), str(rounded)))
        else:
            print("BAD READING - no recover. Reusing last higher reading %s > %s" %
                            (str(last_reading), str(rounded)))
        rounded = last_reading

    return last_reading,rounded

if __name__ == '__main__':
    main(sys.argv[1:])

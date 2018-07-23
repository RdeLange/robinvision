from os import listdir
from os.path import isfile, join, splitext
import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import os
import pickle
import shutil
import base64
import json
import re
from PIL import Image
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import sys
import sched
import time
import datetime
import threading


# Global storage for images
faces_dict = {}

# Create flask app
UPLOAD_FOLDER = '/var/www/html/faces/files'
TEMP_FOLDER = '/root/app'
ENCODINGS_FOLDER = '/root/encodings'
SAVE_UNKNOWN = True
SCHEDULE_ENCODINGS_SAVE = True
SCHEDULE_ENCODINGS_HOUR = 13
SCHEDULE_ENCODINGS_MINUTES = 46
app = Flask(__name__)
app.config['FACES_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['ENCODINGS_FOLDER'] = ENCODINGS_FOLDER
app.config['SAVE_UNKNOWN'] = SAVE_UNKNOWN
app.config['SCHEDULE_ENCODINGS_SAVE'] = SCHEDULE_ENCODINGS_SAVE
app.config['SCHEDULE_ENCODINGS_HOUR'] = SCHEDULE_ENCODINGS_HOUR
app.config['SCHEDULE_ENCODINGS_MINUTES'] = SCHEDULE_ENCODINGS_MINUTES
CORS(app)

# <Picture functions> #


def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        return False, None

    # If none face on the given image was found -> error
    if not faces:
        return False, None
    return True, faces[0]


def get_all_images_files(path):
    #this fuction will load all images in the folderstructure as encodings in memory
    #it will not load the images in the 'Unknown' folder
    knownEncodings = []
    knownNames = []
    image_files = []
    imagePaths = []
    print("[INFO] loading individual images => encode images => save to encodings file...")
    print("[INFO] quantifying faces...")
    # Getting the current work directory (cwd)
    thisdir = path
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            imgpath = os.path.join(r, file)
            if not "/Unknown/" in imgpath:
                if is_picture(file) == True:
                    imagePaths.append(imgpath)
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths))+" - "+ imagePath)
        name = imagePath.split(os.path.sep)[-2]
        success, encoding = calc_face_encoding(imagePath)
        if success == True:
            knownEncodings.append(encoding)
            knownNames.append(name)
    return knownNames,knownEncodings

def save_unknown(input_image, left, top, right, bottom):
    #this function will save faces which are unknown to the current trained system as image in the 'Unknown' folder
    img = Image.open(input_image)
    imagepath = os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown")
    if not os.path.exists(imagepath):
        try:
            os.makedirs(imagepath)
            os.chmod(imagepath, 0o777)
        except OSError:
            return False
            pass
    #create a unique filename based on the current timestamp, including miliseconds
    timestr = dt.now().strftime("%Y%m%d-%H%M%S%f")
    unknownface = img.crop((left, top, right, bottom))
    unknownface.save(os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown", "Unknown_"+timestr+".png"))
    os.chmod(os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown", "Unknown_"+timestr+".png"), 0o777)
    return True


def learn_faces_dict(path):
    #this function will load all encodings and related names of persons via the get_all_images_files function and will save the data in a pickle file
    knownNames, knownEncodings = get_all_images_files(path)
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs", "wb")
    f.write(pickle.dumps(data))
    f.close()
    return data

def get_faces_dict(path):
    #this function will load the encodings and names from the earlier saved file (for fast start of the script)
    #when the file is not available it will fallback by loading all the images individually and save the dataset to a file
    print("[INFO] loading encodings...")
    try:
        data = pickle.loads(open(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs", "rb").read())
        print("[INFO] encodings loaded from file...")
    except:
        data = learn_faces_dict(path)
        print("[INFO] encodings loaded from individual images and saved to encoding file <encodings_db.frs> for accelerate future loading...")
    return data

def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)
    # Get the location as a box of the faces on the image
    face_rects_temp = face_recognition.face_locations(img)
    face_rects = []
    for (i, facerect) in enumerate(face_rects_temp):
        face_rects.append({ "top": face_rects_temp[i][0], "left": face_rects_temp[i][3], "width": face_rects_temp[i][1]-face_rects_temp[i][3],"height": face_rects_temp[i][2]-face_rects_temp[i][0]})
    # now the fun start and we will try to match every face found (uploaded_faces) on the image with the face encodings of our trained system
    faces_found = len(uploaded_faces)
    matches = []
    distances = []
    face_encodings = []
    faces = []
    faces2 = []
    match_encoding = ""
    matchcount = 0
    if faces_found:
        #we build a face_encodings dataset with known faces from our trained system
        for (i, encoding) in enumerate(faces_dict['encodings']):
            face_encodings.append(encoding)
        facecount = 0
        #now we loop for every face on the image (uploaded_face) and check if we can match with face in trained system
        for uploaded_face in uploaded_faces:
            facecount = facecount+1
            #here we do the matching
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face)
            matchcount = 0
            #now we start looping through the matches
            for idx, match in enumerate(match_results):
                if match:
                    matchcount = matchcount +1
                    match = faces_dict['names'][idx]
                    match_encoding = face_encodings[idx]
                    dist = face_recognition.face_distance([match_encoding],
                            uploaded_face)[0]
                    #if we already have found matches before
                    if len(matches) > 0:
                        #if we already found a named match for before we will replace that if the confidence is higher in the new match
                        if match in matches:
                            matchindex = matches.index(match)
                            if distances[matchindex] > dist:
                               distances[matchindex] = dist 
                               faces[matchindex] = {"id":match, "dist": dist} 
                               faces2[matchindex] = {'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0}
                        #if no named matches before we will add the match to our dataset with matches
                        else:
                            faces.append({"id":match, "dist": dist})
                            matches.append(match)
                            distances.append(dist)
                            faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
                    #if this is the first match in our entire dataset we will add the match
                    else:
                        faces.append({"id":match, "dist": dist})
                        matches.append(match)
                        distances.append(dist)
                        faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
            #if no match at all for this face on the image we will create an Unknown entry in our dataset
            if matchcount == 0:
                faces.append({"id":"Unknown", "dist": 0})
                matches.append("Unknown")
                distances.append(0)
                faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': "unknown", 'matched':False,'confidence': int((float((0))*100)+0.5)/100.0})  
                #if we configured our system to save unknowns for future classification we will trigger the function to save this unknown face as image in the 'Unknown' folder
                if app.config['SAVE_UNKNOWN'] == True:
                    result = save_unknown(file_stream, face_rects[facecount-1]['left'], face_rects[facecount-1]['top'], face_rects[facecount-1]['width']+face_rects[facecount-1]['left'], face_rects[facecount-1]['top']+face_rects[facecount-1]['height'])
    response = {'success': True,'facesCount': faces_found,'faces':faces2}
    response_json = json.dumps(response)
    return response_json

# function to get unique names from total trained set of images
# this is needed as we might have several images trained for the same person. This will feedback only the unique names and not all (double) names aligned to images in trained system. list1 (input) is the list with names for every image in the trained dataset. This will contain double names in case of more images trained for the same person
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def remove_person(personname):
    #remove an entire person from the dataset by removing the folder containing the images of this person
    path = os.path.join(os.path.abspath(app.config['FACES_FOLDER']), personname)
    shutil.rmtree(path, ignore_errors=True)
    #remove all entries in dataset for this name
    for (i, name) in enumerate(faces_dict['names']):
       if name == personname:
          faces_dict['names'].pop(i)
          faces_dict['encodings'].pop(i)
    #remove saved pickle file to indicate that new pickle file needs to be created (will be automatically triggered on next restart or per enabled schedule)
    if os.path.exists(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs"):
        os.remove(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs")

# <Picture functions> #

# <Encodings Save Scheduler> #
#below functions are in support of the daily schedule to save the trained system in memory to disk. This will enable a fast load when the system needs to be restarted.

def enable_schedule():
    #this function is triggered when the scheduler is enabled (default or per API). It will set the scheduler as per hour and minutes given (UTC) and will run daily at this time
    #the scheduler function is based on the python sched module
    print("[INFO] Scheduled encodings to disk is enabled")
    daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_HOUR'],app.config['SCHEDULE_ENCODINGS_MINUTES'])
    first_time = dt.combine(dt.now(), daily_time)
    if first_time < dt.now():
        first_time = first_time+ timedelta(days=1)
    print("[INFO] First Scheduled Run will be at " + first_time.strftime("%Y%m%d-%H:%M")+" and will run daily at the same time. All times in UTC!")
    currentevent = scheduler.enterabs(time.mktime(first_time.timetuple()), 1,run_schedule, ("run for the first time",))
    #we should run the scheduler in a thread, otherwise it will block our program as it just 'waits'to execute next scheduled event
    schedulerThread = threading.Thread(name='scheduler_process_thread_',target=scheduler_thread,args=())
    schedulerThread.daemon = False
    schedulerThread.start()
    return "scheduler enabled"

def scheduler_thread():
    scheduler.run(blocking=True)

def disable_schedule():
    #we will stop/disable the scheduler when the disable function is called by the API
    print("[INFO] Scheduled encodings to disk is disabled")
    #only if 1 or more events are available in scheduler
    if scheduler.queue:
        events = 0
        while events <= len(scheduler.queue)-1:
            scheduler.cancel(scheduler.queue[events])
            events+=1
    return "scheduler disabled"

def run_schedule(message):
    print("Scheduled Training Started")
    #as the sched module does not have a default fuction for a daily event, we will continuously have to replan for the next day
    t = dt.combine(dt.now() + datetime.timedelta(days=1), daily_time)
    currentevent = scheduler.enterabs(time.mktime(t.timetuple()), 1, run_schedule, ('Running again',))    
    #now we will do the actual dump of the trained system in memory to disk
    global faces_dict
    names = []
    faces_dict = learn_faces_dict(app.config['FACES_FOLDER'])
    return "Scheduler finished"

# <Controller>

@app.route('/', methods=['POST'])
def web_recognize():
    #check for known faces on the image Posted
    file = request.files['file']
    if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
        return detect_faces_in_image(file)
    else:
        raise BadRequest("Given file is invalid!")

@app.route('/facebox/check', methods=['POST'])
#FACEBOX EMULATOR API TO CHECK AN IMAGE ON KNOWN FACES
#POST DATA SHOULD BE PART OF JSON LIKE {'base64': imagedata}
def web_faceboxemulator():
    r = request
    originimg =base64.b64decode(r.get_json()['base64'])
    with open(app.config['TEMP_FOLDER']+"/temp_upload_image.jpg", 'wb') as f:
        f.write(originimg)
    image2 = open(app.config['TEMP_FOLDER']+"/temp_upload_image.jpg", 'rb')
    result = detect_faces_in_image(image2)
    image2.close()
    return result

@app.route('/train', methods=['GET'])
def web_train():
    #train the system with all the images currently in the folder structure
    print("Training Started")
    global faces_dict
    names = []
    faces_dict = learn_faces_dict(app.config['FACES_FOLDER'])
    for (i, name) in enumerate(faces_dict['names']):
        names.append(name)
    uniquenames = unique(names)
    return jsonify(uniquenames)

@app.route('/addface', methods=['POST'])
def web_addfaces():
    #add a new face by posting the image of the person and providing related personname in parameters
    if 'name' in request.args:
        personname = request.args.get('name').replace(" ", "_")
    elif 'name' in request.form:
        personname = request.form.get('name').replace(" ", "_")
    else:
        raise BadRequest("Name for the face was not given!")
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if not os.path.exists(os.path.join(app.config['FACES_FOLDER'],personname)):
           try:
               os.makedirs(os.path.join(app.config['FACES_FOLDER'],personname))
               os.chmod(os.path.join(app.config['FACES_FOLDER'],personname), 0o777)
           except OSError:
               return False
               pass
        file.save(os.path.join(app.config['FACES_FOLDER'],personname,filename))
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict['names'].append(personname)
            faces_dict['encodings'].append(new_encoding)
            if os.path.exists(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs"):
                os.remove(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs")
        except Exception as exception:
            raise BadRequest(exception)
        file.close()
        os.chmod(os.path.join(app.config['FACES_FOLDER'],personname,filename), 0o777) 
    names = []
    for (i, name) in enumerate(faces_dict['names']):
         names.append(name)
         uniquenames = unique(names)
    return jsonify(uniquenames)

@app.route('/facebox/teach', methods=['POST'])
#FACEBOX EMULATOR TO ADD AN ADDITIONAL IMAGE TO THE DATABASE
def web_faceboxteach():
    if 'name' in request.args:
        personname = request.args.get('name').replace(" ", "_")
    elif 'name' in request.form:
        personname = request.form.get('name').replace(" ", "_")
    else:
        raise BadRequest("Name for the face was not given!")
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if not os.path.exists(os.path.join(app.config['FACES_FOLDER'],personname)):
           try:
               os.makedirs(os.path.join(app.config['FACES_FOLDER'],personname))
               os.chmod(os.path.join(app.config['FACES_FOLDER'],personname), 0o777)
           except OSError:
               return False
               pass
        file.save(os.path.join(app.config['FACES_FOLDER'],personname,filename))
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict['names'].append(personname)
            faces_dict['encodings'].append(new_encoding)
            if os.path.exists(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs"):
                os.remove(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs")
        except Exception as exception:
            raise BadRequest(exception)
        file.close()
        os.chmod(os.path.join(app.config['FACES_FOLDER'],personname,filename), 0o777) 
    feedback = {"success": True}
    return jsonify(feedback)

@app.route('/saveunknown', methods=['POST'])
#ENABLE OR DISABLE SAVING IMAGES OF UNKNOWN PERSONS FOR FUTURE CLASSIFICATION
def web_saveunknown():
    if 'enable' in request.args:
        enable_remember = request.args.get('enable').replace(" ", "_")
    elif 'enable' in request.form:
        enable_remember = request.form.get('enable').replace(" ", "_")
    else:
        raise BadRequest("No valid input given, please specify enable=yes OR enable=no")
    if enable_remember not in ("yes", "no"):
        raise BadRequest("No valid input given, please specify enable=yes OR enable=no")
    if enable_remember == "yes":
        app.config['SAVE_UNKNOWN'] = True
        feedback = {"success": True, "message": "Unknown faces will now be remembered in the Unknown folder and can be accessed via localhost:80"}
    elif enable_remember == "no":
        app.config['SAVE_UNKNOWN'] = False
        feedback = {"success": True, "message": "Unknown faces will no longer be remembered but directly deleted"}
    else:
        feedback = {"success": False, "message": "Something went wrong. Settings have not been changed"}
    return jsonify(feedback)

@app.route('/scheduler', methods=['POST'])
#ENABLE OR DISABLE SCHEDULER TO SAVE ENCODINGS TO DISK AT A GIVEN TIME
def web_scheduler():
    if 'enable' in request.args:
        enable_scheduler = request.args.get('enable').replace(" ", "_")
    elif 'enable' in request.form:
        enable_scheduler = request.form.get('enable').replace(" ", "_")
    else:
        raise BadRequest("No valid input given, please specify enable=yes OR enable=no")
    if enable_scheduler not in ("yes", "no"):
        raise BadRequest("No valid input given, please specify enable=yes OR enable=no")
    if enable_scheduler == "yes":
        if 'hour' in request.args:
            hour_scheduler = int(request.args.get('hour'))
        elif 'hour' in request.form:
            hour_scheduler = int(request.form.get('hour'))
        else:
            raise BadRequest("No valid hour given (please specify between 0 and 23")
        if hour_scheduler not in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23):
            raise BadRequest("No valid time given (please specify between 0 and 23)")
        if 'minutes' in request.args:
            minutes_scheduler = int(request.args.get('minutes'))
        elif 'minutes' in request.form:
            minutes_scheduler = int(request.form.get('minutes'))
        else:
            minutes_scheduler = 0
        if not (0 <= minutes_scheduler <= 59):
            raise BadRequest("No valid time given (please specify between 0 and 59)")
    
    if enable_scheduler == "no":
        app.config['SCHEDULE_ENCODINGS_SAVE'] = False
        disable_schedule()
        feedback = {"succes": True, "message": "Schedule disabled"}
    elif enable_scheduler == "yes":
        app.config['SCHEDULE_ENCODINGS_SAVE'] = True
        app.config['SCHEDULE_ENCODINGS_HOUR'] = hour_scheduler
        app.config['SCHEDULE_ENCODINGS_MINUTES'] = minutes_scheduler
        disable_schedule()
        daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_HOUR'],app.config['SCHEDULE_ENCODINGS_MINUTES'])
        currentevent = ""
        enable_schedule()
        feedback = {"succes": True, "message": "Scheduler enabled to run everyday at "+daily_time.strftime("%H:%M")+ " hours (on a scale of 24). All times in UTC!"}
    else:
        feedback = {"succes": False, "message": "Something went wrong, nothing has been changed in the settings"}
    return jsonify(feedback)

@app.route('/getschedule', methods=['GET'])
def web_getschedule():
    # GET the next scheduled event for saving the trained system in memory to a pickle file on disk for a fast start at next reboot
    events = []
    if request.method == 'GET':
        for (i,event) in enumerate(scheduler.queue):
            events.append({"ScheduleId":i,"ScheduleTime":dt.fromtimestamp(event[0]).strftime("%Y%m%d-%H:%M"),"ScheduleTimeStamp":event[0]})
    return jsonify(events)

@app.route('/faces', methods=['GET'])
def web_faces():
    # GET all the names of the persons part of the trained syste,
    names = []
    print (faces_dict)
    if request.method == 'GET':
        for (i, name) in enumerate(faces_dict['names']):
            print (name)
            names.append(name)
    uniquenames = unique(names)
    return jsonify(uniquenames)
    
@app.route('/removeface', methods=['DELETE'])
def web_removefaces():
    # Remove a person from the database
    names = []
    if 'name' not in request.args:
        raise BadRequest("Identifier for the face was not given!")
    if request.method == 'DELETE':
        remove_person(request.args.get('name'))
    for (i, name) in enumerate(faces_dict['names']):
         names.append(name)
         uniquenames = unique(names)
    return jsonify(uniquenames)

def extract_image(request):
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    return file
# </Controller>


if __name__ == "__main__":
    print("[INFO] Starting by generating encodings for found images...")
    # Calculate known faces
    faces_dict = get_faces_dict(app.config['FACES_FOLDER'])
    #set scheduler for daily saving the trained system in memory to a pickle file on disk for a fast start at reboot
    scheduler = sched.scheduler(time.time, time.sleep)
    daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_HOUR'],app.config['SCHEDULE_ENCODINGS_MINUTES'])
    currentevent = ""
    # Set Scheduler if enabled to save encodings once a day at a given time to disk for faster startup
    if app.config['SCHEDULE_ENCODINGS_SAVE'] == True:
        enable_schedule()    
    else:
        disable_schedule()
    # Start app
    print("[INFO] Starting WebServer...")
app.run(host='0.0.0.0', port=8080, debug=False)

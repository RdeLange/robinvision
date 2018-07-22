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


# Global storage for images
faces_dict = {}

# Create flask app
UPLOAD_FOLDER = '/var/www/html/faces/files'
TEMP_FOLDER = '/root/app'
ENCODINGS_FOLDER = '/root/encodings'
SAVE_UNKNOWN = True
SCHEDULE_ENCODINGS_SAVE = True
SCHEDULE_ENCODINGS_TIME = 2
app = Flask(__name__)
app.config['FACES_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['ENCODINGS_FOLDER'] = ENCODINGS_FOLDER
app.config['SAVE_UNKNOWN'] = SAVE_UNKNOWN
app.config['SCHEDULE_ENCODINGS_SAVE'] = SCHEDULE_ENCODINGS_SAVE
app.config['SCHEDULE_ENCODINGS_TIME'] = SCHEDULE_ENCODINGS_TIME
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
    img = Image.open(input_image)
    imagepath = os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown")
    if not os.path.exists(imagepath):
        try:
            os.makedirs(imagepath)
            os.chmod(imagepath, 0o777)

        except OSError:
            return False
            pass
    timestr = dt.now().strftime("%Y%m%d-%H%M%S%f")
    unknownface = img.crop((left, top, right, bottom))
    unknownface.save(os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown", "Unknown_"+timestr+".png"))
    os.chmod(os.path.join(os.path.abspath(app.config['FACES_FOLDER']), "Unknown", "Unknown_"+timestr+".png"), 0o777)
    return True


def learn_faces_dict(path):
    knownNames, knownEncodings = get_all_images_files(path)
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs", "wb")
    f.write(pickle.dumps(data))
    f.close()
    return data

def get_faces_dict(path):
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
    face_rects_temp = face_recognition.face_locations(img)
    face_rects = []
    for (i, facerect) in enumerate(face_rects_temp):
        face_rects.append({ "top": face_rects_temp[i][0], "left": face_rects_temp[i][3], "width": face_rects_temp[i][1]-face_rects_temp[i][3],"height": face_rects_temp[i][2]-face_rects_temp[i][0]})
    #top, right, bottom, lef
    # Defaults for the result object
    faces_found = len(uploaded_faces)
    matches = []
    distances = []
    face_encodings = []
    faces = []
    faces2 = []
    match_encoding = ""
    matchcount = 0
    if faces_found:
        for (i, encoding) in enumerate(faces_dict['encodings']):
            face_encodings.append(encoding)
        facecount = 0
        for uploaded_face in uploaded_faces:
            facecount = facecount+1
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face)
            matchcount = 0
            for idx, match in enumerate(match_results):
                if match:
                    matchcount = matchcount +1
                    match = faces_dict['names'][idx]
                    match_encoding = face_encodings[idx]
                    dist = face_recognition.face_distance([match_encoding],
                            uploaded_face)[0]
                    if len(matches) > 0:
                        if match in matches:
                            matchindex = matches.index(match)
                            if distances[matchindex] > dist:
                               distances[matchindex] = dist 
                               faces[matchindex] = {"id":match, "dist": dist} 
                               faces2[matchindex] = {'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0}
                        else:
                            faces.append({"id":match, "dist": dist})
                            matches.append(match)
                            distances.append(dist)
                            faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
                    else:
                        faces.append({"id":match, "dist": dist})
                        matches.append(match)
                        distances.append(dist)
                        faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
            if matchcount == 0:
                faces.append({"id":"Unknown", "dist": 0})
                matches.append("Unknown")
                distances.append(0)
                faces2.append({'rect':face_rects[facecount-1], 'id': "dummy.jpg",'name': "unknown", 'matched':False,'confidence': int((float((0))*100)+0.5)/100.0})  
                if app.config['SAVE_UNKNOWN'] == True:
                    result = save_unknown(file_stream, face_rects[facecount-1]['left'], face_rects[facecount-1]['top'], face_rects[facecount-1]['width']+face_rects[facecount-1]['left'], face_rects[facecount-1]['top']+face_rects[facecount-1]['height'])
    response = {'success': True,'facesCount': faces_found,'faces':faces2}
    response_json = json.dumps(response)
    return response_json

# function to get unique names from total trained set of images
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def remove_person(personname):
    path = os.path.join(os.path.abspath(app.config['FACES_FOLDER']), personname)
    shutil.rmtree(path, ignore_errors=True)

# <Picture functions> #

# <Encodings Save Scheduler> #

def enable_schedule():
    print("[INFO] Scheduled encodings to disk is enabled")
    print("[INFO] Schedules encodings will be saved daily at " + str(app.config['SCHEDULE_ENCODINGS_TIME']) + ":00 hours (on 24:00 hours scale)")
    daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_TIME'])
    first_time = dt.combine(dt.now(), daily_time)
    if first_time < dt.now():
        first_time = first_time+ timedelta(days=1)
    print("[INFO] First Scheduled run will be at " + first_time.strftime("%Y%m%d-%H:00"))
    currentevent = scheduler.enterabs(time.mktime(first_time.timetuple()), 1,run_schedule, ("run for the first time",))
    scheduler.run(blocking=False)
    return "scheduler enabled"

def disable_schedule():
    print("[INFO] Scheduled encodings to disk is disabled")
    if currentevent != "":
        scheduler.cancel(currentevent)
    return "scheduler disabled"

def run_schedule(message):
    print("Scheduled Training Started")
    global faces_dict
    names = []
    faces_dict = learn_faces_dict(app.config['FACES_FOLDER'])
    for (i, name) in enumerate(faces_dict['names']):
        names.append(name)
    #uniquenames = unique(names)
    t = dt.combine(dt.now() + datetime.timedelta(days=1), str(daily_time))
    currentevent = scheduler.enterabs(time.mktime(t.timetuple()), 1, run_schedule, ('Running again',))    
    #return jsonify(uniquenames)
    return "Scheduler finished"

# <Controller>

@app.route('/', methods=['POST'])
def web_recognize():
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
#ENABLE OR DISABLE SCHEDULER TO SAVE ENCODINGS TO DISK AT A GIVEN TIME
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
        if 'time' in request.args:
            time_scheduler = int(request.args.get('time'))
        elif 'time' in request.form:
            time_scheduler = int(request.form.get('time'))
        else:
            raise BadRequest("No valid time given (please specify between 0 and 23")
        if time_scheduler not in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23):
            raise BadRequest("No valid time given (please specify between 0 and 23)")
    
    if enable_scheduler == "no":
        app.config['SCHEDULE_ENCODINGS_SAVE'] = False
        disable_schedule()
        feedback = {"succes": True, "message": "Schedule disabled"}
    elif enable_scheduler == "yes":
        app.config['SCHEDULE_ENCODINGS_SAVE'] = True
        app.config['SCHEDULE_ENCODINGS_TIME'] = time_scheduler
        disable_schedule()
        daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_TIME'])
        currentevent = ""
        enable_schedule()
        feedback = {"succes": True, "message": "Scheduler enabled to run everyday at "+str(time_scheduler)+ ":00 hours (on a scale of 24)"}
    else:
        feedback = {"succes": False, "message": "Something went wrong, nothing has been changed in the settings"}
    return jsonify(feedback)

@app.route('/faces', methods=['GET'])
def web_faces():
    # GET
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
    # DELETE
    names = []
    if 'name' not in request.args:
        raise BadRequest("Identifier for the face was not given!")
    if request.method == 'DELETE':
        remove_person(request.args.get('name'))
        for (i, name) in enumerate(faces_dict['names']):
            if name == request.args.get('name'):
               faces_dict['names'].pop(i)
               faces_dict['encodings'].pop(i)
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
    scheduler = sched.scheduler(time.time, time.sleep)
    daily_time = datetime.time(app.config['SCHEDULE_ENCODINGS_TIME'])
    currentevent = ""

    # Set Scheduler if enabled to save encodings once a day at a given time to disk for faster startup
    if app.config['SCHEDULE_ENCODINGS_SAVE'] == True:
        enable_schedule()    
    else:
        disable_schedule()
    # Start app
    print("[INFO] Starting WebServer...")
app.run(host='0.0.0.0', port=8080, debug=False)

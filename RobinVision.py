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


# Global storage for images
faces_dict = {}

# Create flask app
UPLOAD_FOLDER = '/var/www/html/faces/files'
TEMP_FOLDER = '/root/app'
ENCODINGS_FOLDER = '/root/encodings'
app = Flask(__name__)
app.config['FACES_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['ENCODINGS_FOLDER'] = ENCODINGS_FOLDER
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
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]

#############################################################RDL START

def get_all_images_files(path):
    knownEncodings = []
    knownNames = []
    image_files = []
    imagePaths = []
    print("[INFO] quantifying faces...")
    # Getting the current work directory (cwd)
    thisdir = path
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if is_picture(file) == True:
                imagePaths.append(os.path.join(r, file))
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        print(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        #image_files.append([str(i)+"_"+name,imagePath])
        encoding = calc_face_encoding(imagePath)
        knownEncodings.append(encoding)
        knownNames.append(name)
    return knownNames,knownEncodings

#############################################################RDL END

def learn_faces_dict(path):
    knownNames, knownEncodings = get_all_images_files(path)
    
    #print(image_files)
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs", "wb")
    f.write(pickle.dumps(data))
    f.close()
    #faces_dict = data
    #print (data)
    return data

def get_faces_dict(path):
    print("[INFO] loading encodings...")
    try:
        data = pickle.loads(open(app.config['ENCODINGS_FOLDER']+"/encodings_db.frs", "rb").read())
    except:
        data = learn_faces_dict(path)
    return data

def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    
    #try:
    #    img = np.fromstring(file_stream, dtype=np.uint8)
    #except:
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)

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
        for uploaded_face in uploaded_faces:
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face)
            matchcount = 0
            for idx, match in enumerate(match_results):
                if match:
                    matchcount = matchcount +1
                    #match = list(faces_dict['names'][idx])
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
                               faces2[matchindex] = {'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0}
                        else:
                            faces.append({"id":match, "dist": dist})
                            matches.append(match)
                            distances.append(dist)
                            faces2.append({'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
                    else:
                        faces.append({"id":match, "dist": dist})
                        matches.append(match)
                        distances.append(dist)
                        faces2.append({'name': match, 'matched':True,'confidence': int((float((1-dist))*100)+0.5)/100.0})
            if matchcount == 0:
                faces.append({"id":"Unknown", "dist": 0})
                matches.append("Unknown")
                distances.append(0)
                faces2.append({'name': "unknown", 'matched':False,'confidence': int((float((0))*100)+0.5)/100.0})  
    
    response = {'success': True,'facesCount': faces_found,'faces':faces2}
    response_json = json.dumps(response)
    #data = {"count": faces_found, "faces": faces}
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

# <Controller>


@app.route('/', methods=['POST'])
def web_recognize():
    file = request.files['file']
    #file = extract_image(request)

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
    #print(r.get_json())
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
    print(faces_dict)
    faces_dict = learn_faces_dict(app.config['FACES_FOLDER'])
    for (i, name) in enumerate(faces_dict['names']):
        names.append(name)
    uniquenames = unique(names)
    print(uniquenames)
    return jsonify(uniquenames)

@app.route('/addface', methods=['POST'])
def web_addfaces():
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'POST':
        image = request.files['image']
        #try:
        #    new_encoding = calc_face_encoding(image)
        #    faces_dict['names'].append(request.args.get('id'))
        #    faces_dict['encodings'].append(new_encoding)
        #except Exception as exception:
        #    raise BadRequest(exception)

        filename = secure_filename(image.filename)
        if not os.path.exists(os.path.join(app.config['FACES_FOLDER'],request.args.get('id'))):
           try:
               os.makedirs(os.path.join(app.config['FACES_FOLDER'],request.args.get('id')))
           except OSError:
               return False
               pass
        #contents = image.read()
        # do something with file contents
        image.save(os.path.join(app.config['FACES_FOLDER'],request.args.get('id'),filename))
       
        
        try:
            new_encoding = calc_face_encoding(image)
            faces_dict['names'].append(request.args.get('id'))
            faces_dict['encodings'].append(new_encoding)
        except Exception as exception:
            raise BadRequest(exception)
        image.close()

    names = []
    for (i, name) in enumerate(faces_dict['names']):
         names.append(name)
         uniquenames = unique(names)

    return jsonify(uniquenames)


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
    
    names = []
    
    # DELETE
    
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'DELETE':
        remove_person(request.args.get('id'))
        #faces_dict.pop(request.args.get('id'))
    
        for (i, name) in enumerate(faces_dict['names']):
            if name == request.args.get('id'):
               faces_dict['names'].pop(i)
               faces_dict['encodings'].pop(i)
    names = []
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
    print(faces_dict)

    # Start app
    print("[INFO] Starting WebServer...")
app.run(host='0.0.0.0', port=8080, debug=False)

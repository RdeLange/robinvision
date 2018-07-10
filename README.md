# Robin Vision - Docker image

This project provides a docker image which offers a web service to recognize known faces on images. It's based on the great [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) and [JanLoebel/face_recognition](https://github.com/JanLoebel/face_recognition) projects and just add additional web services using the Python `face_recognition`-library. This service also includes a FaceBox emulate API I create to use the FaceBox component in Home Assistant without having the limitations of the free FaceBox docker container.

On top of that I included a slightly adjusted KCFinder implementation so you can manage your face images via a browser and trigger the 'learn faces API.

## Get started

### Build the Docker image

Start by building the docker image with a defined name. This can take a while.

```bash
docker build -t robinvision .
```

### Run the Docker image

Start the image and forward port 8080 & 80.

```bash
docker run -d -p 8080:8080 -p 80:80 robinvision
```

## Features

### Register known faces

Simple `POST` an image-file to the `/addface` endpoint and provide an identifier.
`curl -X POST -F "file=@person1.jpg" http://localhost:8080/addface?id=person1`

### Read registered faces

Simple `GET` the `/faces` endpoint.
`curl http://localhost:8080/faces`

### Identify faces on image

Simple `POST` an image-file to the web service.
`curl -X POST -F "file=@person1.jpg" http://localhost:8080/`

### Delete persons
Simple `DELETE` a person from the web service
`curl -X DELETE http://localhost:8080/id=person1`

### Train the system

Simple `GET` the `/train` endpoint.
`curl http://localhost:8080/train`

### FaceBox emulation

In order to be able to make use of this service from the excelent Home Assistant software I have created an API which emulates the FaceBox docker container API. 
Just setup the FaceBox component in Home Assistant as per Home Assistant documentation, use the ip address of the RobinVision container as the ip address and the port is 8080. Have Fun
For reference, the API endpoint is /facebox/check. It will only emulates the base64 json implementation (as used in the Home Assistant component)



### Web Interface

I have added a webinterface to manage your images and to trigger the training of the system (after adding or removing imgages/persons)
The interface is based on KCFinder. The images are managed in folders under the `files` main folder. Every folder represents a person, the name of the person is the folder name. In the folders you can add/delete images of that specific person. After any change in the person/image database, please make sure to push the train button to get the system retrained. The system will retrain itself after any system (container) relaunch by default.


## Notes

I'm not a programming Guru. Code might be not fully optimised. But it's working -:)
Enjoy!

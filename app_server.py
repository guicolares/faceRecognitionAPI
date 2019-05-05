#!/usr/bin/env python
import face_recognition
import numpy as np
from flask import Flask, jsonify, request, redirect
from signal import signal, SIGPIPE, SIG_DFL
import base64
import cv2
from PIL import Image
from StringIO import StringIO

app = Flask(__name__)

knownFaceEncodings = []
knownFaceNames = []

processThisImage = True

def readb64(base64String):
	sbuf = StringIO()
	sbuf.write(base64.b64decode(base64String))
	pimg = Image.open(sbuf)
	image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

	# Resize frame of video to 1/4 size for faster face recognition processing
	smallFrame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgbSmallFrame = smallFrame[:, :, ::-1]
	return rgbSmallFrame

@app.route('/upload-face', methods=['POST', 'GET'])
def uploadFace():
	if request.method == 'POST':
		# use for mobile upload
		file = request.form['file']
		image = readb64(file)
		name = request.form['name']
		faceLocations = face_recognition.face_locations(image)
		if len(faceLocations) > 0:
			faceEncodings = face_recognition.face_encodings(image, faceLocations)
			if len(faceEncodings) > 0:
				knownImageEncondig = faceEncodings[0] #get first face detected
				knownFaceEncodings.append(knownImageEncondig)
				knownFaceNames.append(name)
				return jsonify({"face_uploaded": True, "face_found": True})
			else:
				return jsonify({"face_uploaded": False, "face_found": True})
		else:
			return jsonify({"face_found": False, "face_uploaded": False})

		
	else:
			return '''
	<!doctype html>
	<title>Selfie?</title>
	<h1>Upload a picture!</h1>
	<form method="POST" enctype="multipart/form-data">
	  <input type="file" name="file">
	  <input type="text" name="name"> 
	  <input type="submit" value="Upload">
	</form>
	'''
@app.route('/detect-face', methods=['POST', 'GET'])
def detectFace():
	if request.method == 'POST': 
		file = request.form['file']
		image = readb64(file)
    
		faceNames = []
		if processThisImage:

			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(image)
			faceEncodings = face_recognition.face_encodings(image, face_locations)
			for faceEncoding in faceEncodings:
				matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
				name = "Unknown"
				faceDistances = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
				bestMatchIndex = np.argmin(faceDistances)

				if matches[bestMatchIndex]:
					name = knownFaceNames[bestMatchIndex]
					
			faceNames.append(name)

		# processThisImage = not processThisImage

		result = {"face_names": faceNames }
		return jsonify(result)
	else: 
		return '''
	<!doctype html>
	<title>Selfie?</title>
	<h1>Upload a picture!</h1>
	<form method="POST" enctype="multipart/form-data">
	  <input type="file" name="file">
	  <input type="submit" value="Upload">
	</form>
	'''
		# return jsonify({"file_found": False})


if __name__ == '__main__':
	signal(SIGPIPE, SIG_DFL)
	app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)	


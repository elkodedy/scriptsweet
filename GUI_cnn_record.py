import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import time
import cv2
import os
import PySimpleGUI as sg
import datetime
from playsound import playsound


def main():
    sg.theme('Reddit')

    # define layout
    layout_status = [
        [sg.Text('Menggunakan Masker : ', justification='left', font='Helvetica 10')],
        [sg.Text('0 Orang', key='mask-count', justification='left', font='Helvetica 20')],
        [sg.Text('Tidak Menggunakan Masker : ', justification='left', font='Helvetica 10')],
        [sg.Text('0 Orang', key='no-mask-count', justification='left', font='Helvetica 20')]]
    layout = [
                [sg.Text('Sistem Pendeteksi Penggunaan Masker Otomatis \n Convolutional Neural Network', justification='center', font='Helvetica 20', background_color='#d4d4d4', expand_x=True)],
                [
                    [sg.Image(filename='wp.png', size=(640, 360), key='image'),
                     sg.Frame(layout=layout_status, title='', border_width=0)],
                ],
                [
                    sg.Button('Play', size=(10, 1), font='Helvetica 14'),
                    sg.Button('Capture', size=(10, 1), font='Helvetica 14'),
                    sg.Button('Record', size=(10, 1), font='Helvetica 14'),
                    sg.Button('Stop', size=(10, 1), font='Any 14'),
                    sg.Button('Exit', size=(10, 1), font='Helvetica 14')
                ],
                [
                    sg.Text('Jurusan Teknik Informatika, Fakultas Teknik, Universitas Halu Oleo', justification='c', font='Helvetica 10', background_color='#d4d4d4', expand_x=True)
                ]
            ]

    # show layout
    window = sg.Window('CCTV Face Mask Detection With CNN', layout, location=(0, 0), element_justification='c', margins=(0,0))
    # window = sg.Window('CCTV Face Mask Detection Caffe-CNN', layout, location=(325, 100), element_justification='c', margins=(0,0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    # cap = cv2.VideoCapture(0)
    recording = True
    playing = False

    vid_recording = False

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            record_output.release()
            return

        elif event == 'Play':
            playing = True

        elif event == 'Record':
            vid_time = time.time()

            # initializing video recorder
            vid_cod = cv2.VideoWriter_fourcc(*'XVID')
            vid_path = "records/records " + str(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")) + ".mp4"
            # record_output = cv2.VideoWriter(vid_path, vid_cod, 7.0, (640, 480))
            record_output = cv2.VideoWriter(vid_path, vid_cod, 7.0, (960, 540))

            vid_recording = True
            count_frame = 1

        elif event == 'Stop':
            playing = False
            # load wallpaper then convert it to
            img_wp = "wp.png"
            # window['image'].update(filename=img_wp, size=(640, 480))
            # window['image'].update(filename=img_wp, size=(960, 720))

        if playing:
            fps_time = time.time()
            # get frame then resize it
            frame = vs.read()
            # print(type(frame))
            # frame = imutils.resize(frame, width=640)
            frame = imutils.resize(frame, width=960)

            # save real frame
            real_frame = frame
            if event == 'Capture':
                imname = "capture/caffe_cnn/" + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".png"
                cv2.imwrite(imname, real_frame)

            # detect faces and mask
            (locs, preds) = my_mask_detector(frame, face_detector, mask_model)

            # loop detections
            mask_count, no_mask_count = 0, 0
            for (box, pred) in zip(locs, preds):
                # get coordinat box and predictions
                (start_x, start_y, end_x, end_y) = box
                (mask, without_mask) = pred

                # initialize label and color
                label = "Mask" if mask > without_mask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                if mask > without_mask:
                    mask_count = mask_count + 1
                else:
                    no_mask_count = no_mask_count + 1

                # add probability
                # label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

                # display label
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
                cv2.rectangle(frame, (start_x, start_y - 40), (end_x, start_y), color, -1)
                cv2.putText(frame, label, (start_x, start_y - 10), font, 0.5, (255, 255, 255), thickness)

            if (len(preds) == 0):
                (h, w) = frame.shape[:2]
                cv2.putText(frame, "No face found...", (30, h - 30), font, 0.6, (255, 255, 255), thickness, cv2.LINE_AA)


            # ret, frame = cap.read()
            window['mask-count'].update((str(mask_count) + " Orang"))
            window['no-mask-count'].update((str(no_mask_count) + " Orang"))
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

            # save detect frame
            if event == 'Capture':
                imname_res = "capture/caffe_cnn/" + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + " (result).jpg"
                cv2.imwrite(imname_res, frame)
                # playsound('mixkit-camera-long-shutter-1431.wav')
            # save detect frame

            if vid_recording:
                record_output.write(frame)
                print("[Recording]", count_frame)
                count_frame = count_frame + 1
                if (time.time() - vid_time) > 5.5:
                    vid_recording = False


            print("FPS: ", 1.0 / (time.time() - fps_time))  # FPS = 1 / time to process loop
            # playsound('mixkit-camera-long-shutter-1431.wav')

        # k = cv2.waitKey(25) & 0xff
        # # if k == 27:
        # #     break
        # if k == ord("q"):
        #     print("RETURN")
        #     return
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def my_mask_detector(frame, face_detector, mask_model):
    # get height and width
    (h, w) = frame.shape[:2]
    # print (h, " ", w)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # deteksi face dari blob
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # initialize list
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence/probability
        confidence = detections[0, 0, i, 2]
        # print(detections[0, 0, i, 2])
        # filter out weak detections
        if confidence > 0.35:
            # initialize (x, y)-coordinates for bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # ensure the bounding boxes in correct place
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # get face, turn it to RGB, resize it then preprocess it
            face = frame[start_y:end_y, start_x:end_x]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (150, 150))
                # face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                # cv2.imshow("face", face)

                # add face and location to list
                faces.append(face)
                locs.append((start_x, start_y, end_x, end_y))
                # print(type(face))

    # if face detected
    if len(faces) > 0:
        # turn faces to numpy array then predict mask
        faces = np.array(faces, dtype="float32")
        preds = mask_model.predict(faces, batch_size=32)

    # return face locations and predictions
    return (locs, preds)

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
font_scale = 1

# load face detector model
print("[INFO] loading face detector model...")
prototxt_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
weights_path = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# load the mask detector model
print("[INFO] loading face mask detector model...")
mask_model = load_model("my_mask_model_2000")

# initialize the stream
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(src="rtsp://admin:Admin17_@192.168.1.24:554/H.264").start()

# response = requests.get('http://192.168.1.24')
# if response.status_code == 200:
#     print('Web site exists')
#     vs = VideoStream(src="rtsp://admin:Admin17_@192.168.1.24:554/H.264").start()
# else:
#     vs = VideoStream(src=0).start()
#     print('Web site does not exist')

time.sleep(2.0)
# loop over the frames from the video stream
# ------------------- GUI ------------------
main()

import face_recognition
import cv2



webcam = cv2.VideoCapture(0)

face_locations = []
while True:
    ret, bgr_frame = webcam.read()
    rgb_frame = bgr_frame[:,:,::-1] #reverse bgr_frame
    small_frame_rgb = cv2.resize(rgb_frame, (0,0), fx=0.25, fy=0.25) # resize for further performance optimization
    face_locations = face_recognition.face_locations(small_frame_rgb)

    # draw results on every frame, if counter %4 != 4 draw old result
    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        print(top, bottom)
        # scale the rectangle to get the whole face with a small buffer
        big_top = round(top - 0.45 * (bottom - top))
        big_bottom = round(bottom + 0.05 * (bottom - top))
        big_right = round(right + 0.25 * (right - left))
        big_left = round(left - 0.25 * (right - left))

        face_subimage = bgr_frame[big_top:big_bottom, big_left:big_right]
        face_subimage = cv2.resize(face_subimage, (128, 128))
        cv2.imwrite("extracted_face.jpg", face_subimage)

        # cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0,0,255), 5) # draw red rectangle arround face
        cv2.rectangle(bgr_frame, (big_left, big_top), (big_right, big_bottom), (0,0,255), 5) # draw second bigger green rectangle arround face


    cv2.imshow('Webcam', bgr_frame)

    #  hit 'q' to quit
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break



webcam.release()
cv2.destroyAllWindows()

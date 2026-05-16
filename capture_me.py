import cv2
import numpy as np
from pathlib import Path
import face_recognition
import tqdm
import time

def capture_me(total: int=600, interval_sec: float=0.5, dst_dir: Path=Path("data/me")):
    """
    starts webcam and takes pictures every interval_sec and saves extracted face in dst_dir if face was found until total number of pictures is reached
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    webcam = cv2.VideoCapture(0) # access default webcam
    
    print(f"Capturing a total of {total} picutres, capturing every {interval_sec}s and saving in {dst_dir}")
    
    count = 0
    interrupted = False
    pbar = tqdm.tqdm(total=total, desc="Capturing faces")
    t1 = time.time()
    while count < total:
        t2 = time.time()
        ret, bgr_frame = webcam.read()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)#reverse bgr_frame
        small_frame_rgb = cv2.resize(rgb_frame, (0,0), fx=0.25, fy=0.25) # resize for further performance optimization
        face_locations = face_recognition.face_locations(small_frame_rgb)
        
        big_height, big_width = bgr_frame.shape[:2]

        # draw results on every frame, if counter %4 != 4 draw old result
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # scale the rectangle to get the whole face with a small buffer
            big_top = round(top - 0.45 * (bottom - top))
            big_bottom = round(bottom + 0.05 * (bottom - top))
            big_right = round(right + 0.25 * (right - left))
            big_left = round(left - 0.25 * (right - left))

            # out of bounds check
            if (big_top < 0 or big_left < 0 or big_bottom > big_height or big_right > big_width):
                continue
            if big_bottom <= big_top or big_right <= big_left:
                continue
            if t2 - t1 >= interval_sec:
                face_subimage = bgr_frame[big_top:big_bottom, big_left:big_right]
                face_subimage = cv2.resize(face_subimage, (128, 128))
                cv2.imwrite(str(dst_dir / f"me_{count}.jpg"), face_subimage)
                t1 = t2
                count += 1
                pbar.update(1)

            cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0,0,255), 5) # draw red rectangle arround face
            cv2.rectangle(bgr_frame, (big_left, big_top), (big_right, big_bottom), (0,255, 0), 5) # draw second bigger green rectangle arround face


        cv2.imshow('Webcam', bgr_frame)

        #  hit 'q' to quit
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            if count < total:
                interrupted = True
            break


    pbar.close()
    webcam.release()
    cv2.destroyAllWindows()
    if interrupted:
        print(f"Interrupted. Captured {count} images.")
    else:
        print(f"Done. Captured {count} images.")

if __name__ == "__main__":
    capture_me()

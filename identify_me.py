import face_recognition
import cv2
import numpy as np
from pathlib import Path
from calculate_eigenfaces import project_faces
from reconstruct_face import identify_faces, reconstruct_face




def identify_known_persons(face_images: list[np.ndarray],
                           training_coeffs: np.ndarray,
                           known_persons: dict[str, np.ndarray],
                           pcs: np.ndarray,
                           mean_data: np.ndarray,
                           threshold: float= 0.3
                           ) ->  tuple[list[str], list[int], np.ndarray]:
    """
    Identify known persons in a list of extracted face images:
    Project face images into eigenspace
    Calculate angles to all gallery-coefficients
    Find closest matches
    Classify:
        If closest match is labled as a known person
        and
        angle < threshold
        -> Extracted face is classified with lable

        Else face is classified as unknown

    Arguments:
    face_images: list of grayscale 128x128 face images
    training_coeffs: coefficients of known faces (training and me)
    known_persons: dictionary, key: name of known person, value: corresponding coefficient matrix
    pcs: principle components (eigenfaces of) of training data
    mean_data: mean of training faces
    threshold: threshold for angle of extracted face

    Return:
    lables: list of strings, either a name of a person or "unknown"
    best_indices: list of integers, indices of closest match
    coeffs_test: coefficients of face_images
    """
    n_training = len(training_coeffs)
    gallery_coeffs = training_coeffs.copy()
    person_index_ranges = {}
    cur_idx = n_training
    for name, coeffs in known_persons.items():
        person_index_ranges[name] = range(cur_idx, cur_idx + len(coeffs))
        gallery_coeffs = np.vstack([gallery_coeffs, coeffs])
        cur_idx += len(coeffs)

    scores, _, coeffs_test= identify_faces(gallery_coeffs, pcs, mean_data, imgs_test=face_images)
    lables = []
    best_indices = []
    for j in range(len(face_images)):
        best_idx = np.argmin(scores[:,j])
        best_indices.append(best_idx)
        best_angle = scores[best_idx, j]

        lable = "unknown"
        if best_angle < threshold:
            for name, person_index_range in person_index_ranges.items():
                if best_idx in person_index_range:
                    lable = name
                    break
        lables.append(lable)

    return lables, best_indices, coeffs_test



def identify_persons_in_webcam(training_coeffs: np.ndarray,
                               known_persons: dict[str, np.ndarray],
                               pcs: np.ndarray,
                               mean_data: np.ndarray,
                               threshold: float= 0.3
                               ):
    """
    Access webcam, find faces in frame and identify persons
    """
    webcam = cv2.VideoCapture(0)

    face_locations = []
    while True:
        ret, bgr_frame = webcam.read()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)#reverse bgr_frame
        small_frame_rgb = cv2.resize(rgb_frame, (0,0), fx=0.25, fy=0.25) # resize for further performance optimization
        face_locations = face_recognition.face_locations(small_frame_rgb)
        
        big_height, big_width = bgr_frame.shape[:2]

        face_images = []
        valid_face_locations = []
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

            face_subimage = bgr_frame[big_top:big_bottom, big_left:big_right]
            face_subimage = cv2.resize(face_subimage, (128, 128))
            face_subimage = cv2.cvtColor(face_subimage, cv2.COLOR_BGR2GRAY)
            face_images.append(face_subimage)
            valid_face_locations.append((top, right, bottom, left))


        lables, best_indices, coeffs_test = identify_known_persons(face_images, training_coeffs, known_persons, pcs, mean_data, threshold=threshold)

        #show second window with extracted image, reconstructed face from eigenfaces, and closest match
        if len(face_images) > 0:
            h, w = face_images[0].shape
            panels = []
            for j, (face, best_idx) in enumerate(zip(face_images, best_indices)):
                # Rekonstruktion
                reconstruction = reconstruct_face(coeffs_test[j], pcs, mean_data, pcs.shape[0])
                reconstruction_img = cv2.normalize(
                    reconstruction.reshape(h, w), None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

                # Closest match laden
                gallery_paths = [Path(p) for p in np.load("data/eigenfaces/gallery_paths.npy")]
                match_img = cv2.imread(str(gallery_paths[best_idx]), cv2.IMREAD_GRAYSCALE)
                match_img = cv2.resize(match_img, (w, h))

                panel = np.hstack([face, reconstruction_img, match_img])
                panels.append(panel)

            debug_frame = np.vstack(panels)
            cv2.imshow("Extracted | Reconstruction | Closest Match", debug_frame)
            cv2.moveWindow("Extracted | Reconstruction | Closest Match", 0, 600)


        for idx, (top, right, bottom, left) in enumerate(valid_face_locations):
            # scale the rectangle to get the whole face with a small buffer
            big_top = round(top - 0.45 * (bottom - top))
            big_bottom = round(bottom + 0.05 * (bottom - top))
            big_right = round(right + 0.25 * (right - left))
            big_left = round(left - 0.25 * (right - left))

            lable = lables[idx]
            font = cv2.FONT_HERSHEY_DUPLEX
            if lable == "unknown":
                cv2.rectangle(bgr_frame, (big_left, big_top), (big_right, big_bottom), (0, 0, 255), 5)
                cv2.rectangle(bgr_frame, (big_left, big_bottom + 35), (big_right, big_bottom), (0, 0, 255), 5)
                cv2.rectangle(bgr_frame, (big_left - 3, big_bottom + 35 ), (big_right + 3, big_bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(bgr_frame, lable, (big_left + 6, big_bottom + 35 - 6), font, 1.0, (1, 1, 1), 1)
            else:
                cv2.rectangle(bgr_frame, (big_left, big_top), (big_right, big_bottom), (0, 255, 0), 5)
                cv2.rectangle(bgr_frame, (big_left, big_bottom + 35), (big_right, big_bottom), (0, 255, 0), 5)
                cv2.rectangle(bgr_frame, (big_left - 3, big_bottom + 35 ), (big_right + 3, big_bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(bgr_frame, lable, (big_left + 6, big_bottom + 35 - 6), font, 1.0, (0, 0, 0), 1)

        #  hit 'q' to quit
        cv2.imshow('Webcam', bgr_frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    training_coeffs = np.load("data/eigenfaces/coeffs_train.npy")
    pcs = np.load("data/eigenfaces/pcs.npy")
    mean_data = np.load("data/eigenfaces/mean.npy")
    kolja = np.load("data/eigenfaces/coeffs_me.npy")
    know_persons = {}
    know_persons["Kolja"] = kolja
    threshold = .7

    identify_persons_in_webcam(training_coeffs, know_persons, pcs, mean_data, threshold=threshold)


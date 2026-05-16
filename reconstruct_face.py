from pathlib import Path
import numpy as np
import cv2
from calculate_eigenfaces import load_images, project_faces
import random
import sys


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray,
                   path_test: Path=None, imgs_test: list=None) -> (np.ndarray, list[np.ndarray], np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data, can be None
    imgs_test: list of test images, can be None, but at least one should be not None


    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    if imgs_test is None and path_test is not None:
        imgs_test = load_images(path_test)
    elif imgs_test is None:
        raise ValueError("either path_test or imgs_test should be not None")

    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    test_num = coeffs_test.shape[0]
    train_num = coeffs_train.shape[0]
    scores = np.zeros((train_num, test_num))

    #  Iterate over all images and calculate pairwise correlation
    for i, train in enumerate(coeffs_train):
        for j, test in enumerate(coeffs_test):
            scores[i, j] = np.arccos(np.dot(test, train)/(np.linalg.norm(train)*np.linalg.norm(test)))

    return scores, imgs_test, coeffs_test



def reconstruct_face(coeffs: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruct a face from its PCA coefficient using k eigenfaces

    Arguments:
    coeffs: row vector containing the first k PCA coefficients
    pcs: n x k matrix containing principle components (eigenfaces)
    mean_data: mean data that was subtracted before computation of SVD/PCA
    k: number of eigenfaces used for reconstruction

    Return:
    reconstruction: row vector of length n containing the reconstructed face
    """
    reconstruction = mean_data.copy()
    for i in range(k):
        reconstruction += coeffs[i] * pcs[i]
    return reconstruction

def run(gallery_coeffs: np.ndarray,
        pcs: np.ndarray,
        mean_data: np.ndarray, 
        gallery_path: Path,
        test_path: Path
        ):
    """
    Arguments:
    gallery: coefficients of known faces
    pcs: princaple components of training data
    mean_data: mean of training data
    coeffs_train: coefficients of training data
    test_path: directory with test images
    """
    test_images = [Path(p) for p in np.load(str(test_path))]
    gallery     = [Path(p) for p in np.load(str(gallery_path))]
    print("selecting random image")
    image_path = random.choice(test_images)


    print(f"selected {image_path.name}")
    img_test = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = img_test.shape
    print("identifying face")
    scores, imgs_test, coeffs_test = identify_faces(gallery_coeffs, pcs, mean_data, imgs_test=[img_test])
    best_idx = np.argmin(scores[:,0])
    print(f"best_idx = {best_idx}")
    print(f"Closest match: {gallery[best_idx].name} (angle: {np.degrees(scores[best_idx, 0]):.2f}°)")

    print("reconstructing image")
    k = pcs.shape[0]
    reconstruction = reconstruct_face(coeffs_test[0], pcs, mean_data, k)
    reconstruction_img = reconstruction.reshape(h,w)
    reconstruction_img = cv2.normalize(reconstruction_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    match_img = cv2.imread(str(gallery[best_idx]), cv2.IMREAD_GRAYSCALE)
    combined = np.hstack([img_test, reconstruction_img, match_img])
    cv2.imshow("Original | Reconstruction | Closest Match", combined)
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
    





if __name__ == "__main__":
    gallery = np.load("data/eigenfaces/coeffs_gallery.npy")
    pcs = np.load("data/eigenfaces/pcs.npy")
    mean_data = np.load("data/eigenfaces/mean.npy")
    gallery_paths = Path("data/eigenfaces/gallery_paths.npy")
    test_paths = Path("data/eigenfaces/me_paths.npy")
    while(True):
        run(gallery, pcs, mean_data, gallery_paths, test_paths)

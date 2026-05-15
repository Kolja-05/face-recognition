import enum
from pathlib import Path
import re
import numpy as np
import cv2


def project_faces(pcs: np.ndarray, images: list[np.ndarray], mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis

    Arguments:
    pcs: matrix containing principle components / eigenfunctions as row vectors
    images: list of original images from which pcs were created as np.ndarray
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return: 
    coefficients: basis functions for input images, each row contains coefficients of one image
    """
    n = len(images)
    k = pcs.shape[0]
    coefficients = np.zeros((n,k))
    for i, img, in enumerate(images):
        img = img.flatten() - mean_data
        coefficients[i] = np.dot(pcs, img)
    
    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: Path) -> (np.ndarray, list[np.ndarray], np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    imgs_test = load_images(path_test)[0]

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

def main():
    train_dir = Path("data/clean")
    print(f"loading images")
    images = load_images(train_dir)
    h, w = images[0].shape
    
    print(f"setting up Data matrix")
    D = setup_data_matrix(images)
    
    print(f"calculating principle components")
    pcs, svals, mean_data = calculate_pca(D)
    
    print(f"analysing spectrum")
    k = analyse_spectrum(svals, threshold=0.8)

    print(f"using k={k} eigenfaces")
    pcs_k = pcs[:k]

    print("projecting images")
    coeffs = project_faces(pcs_k, images, mean_data)

    idx = np.random.randint(len(images))
    original = images[idx]
    coeff = coeffs[idx]

    print(f"reconstructing image")
    reconstruction = reconstruct_face(coeffs, pcs_k, mean_data, k)
    reconstruction.reshape(h,w)

    cv2.imshow("Original", original.astype(np.uint8))
    cv2.imshow("Reconstruction", np.clip(reconstruction, 0, 255).astype(np.uint8))

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

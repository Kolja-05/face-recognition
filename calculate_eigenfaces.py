from pathlib import Path
import numpy as np
import cv2
import time



def load_images(src_path: Path, image_extensions = {".jpg", ".jpeg", ".png"}) -> list:
    """
    load all images in path that have given file file_ending
    
    Arguments:
    path: path of directory containing image files
    image_extensions: string that image files have to end with

    Return:
    images: list of grayscale images as numpy.ndarray (dtype float64)
    """
    images = []

    for path in sorted(src_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        images.append(gray)

    return images


def setup_data_matrix(images: list[np.ndarray]) -> np.ndarray:
    """
    Create data matrix out of list of images

    Arguments:
    images: list of images (assumed to be all hoogeneous in size)

    Return:
    D: data matrix that contains flattened images as rows
    """
    h = images[0].shape[0]
    w = images[0].shape[1]
    
    n = h * w
    m = len(images)
    # D is an n x m Matrix
    D = np.zeros((m, n))
    for i, img in enumerate(images):
        D[i] = img.flatten()

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principle component analysis (pca) for given data matrix D

    Arguments:
    D: data matrix containing images as row vectors

    Return:
    pcs: matrix containing primary components
    svals: singular valus associated with principle components
    mean_daata: mean that was subtracted from data
    """
    mean_data = np.mean(D, axis=0)
    D[:] = D[:] - mean_data

    U, svals, pcs = np.linalg.svd(D, full_matrices=False)
    
    return pcs, svals, mean_data




def analyse_spectrum(svals: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values
    is contained in first k singular vectors

    Arguments:
    svals: vector containing singular values
    threshold: threshold for determining k
    """
    svals = (1/np.linalg.norm(svals)) * svals
    k = 0
    sum = 0
    total = np.sum(svals)
    while sum < threshold*total:
        sum += svals[k]
        k += 1
    return k

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




def main():
    train_dir = Path("data/clean")
    print(f"creating eigenfaces directory")
    dst_dir= Path("data/eigenfaces")
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_paths = sorted([p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:4000]
    np.save(dst_dir / "train_paths.npy", np.array([str(p) for p in all_paths]))

    print(f"loading images")
    images = load_images(train_dir)[:4000]
    h, w = images[0].shape
    
    print(f"setting up Data matrix")
    D = setup_data_matrix(images)
    t0 = time.perf_counter()
    print(f"calculating principle components")
    pcs, svals, mean_data = calculate_pca(D)
    t1 = time.perf_counter()
    print(f"PCA time: {t1-t0:.4f} seconds")
    
    print(f"analysing spectrum")
    k = analyse_spectrum(svals, threshold=0.8)
    
    pcs_k = pcs[:k]
    #save eigenfaces as numerical data
    print(f"saving eigenfaces as numerical data")
    np.save(dst_dir / "pcs.npy", pcs_k)
    np.save(dst_dir / "mean.npy", mean_data)
    np.save(dst_dir / "svals.npy", svals)
    coeffs_train = project_faces(pcs_k, images, mean_data)
    np.save(dst_dir / "coeffs_train.npy", coeffs_train)
    all_paths = sorted([p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:4000]

    #save eigenfaces as images
    n, w = images[0].shape
    for i in range(k):
        ef = pcs_k[i].reshape(h,w)
        ef_norm = cv2.normalize(ef, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(dst_dir / f"eigenface_{i:03d}.png"), ef_norm.astype(np.uint8))

    print("successfully saved eigenfaces")
        

if __name__ == "__main__":
    main()


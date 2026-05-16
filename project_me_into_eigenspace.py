from calculate_eigenfaces import load_images, project_faces
import numpy as np
import cv2
from pathlib import Path

def project_me_into_eigenbasis():
    dst_dir = Path("data/eigenfaces")
    me_dir = Path("data/me")

    pcs = np.load(dst_dir / "pcs.npy")
    mean_data = np.load(dst_dir / "mean.npy")

    print("pojecting me into eigenspace")

    me_paths = sorted([p for p in me_dir.iterdir() if p.suffix.lower() in {".jpg", "jpeg", "png"}])
    images_me = load_images(me_dir)
    coeffs_me = project_faces(pcs, images_me, mean_data)

    coeffs_train = np.load(dst_dir / "coeffs_train.npy")
    
    train_paths = np.load(dst_dir / "train_paths.npy")

    coeffs_gallery = np.vstack([coeffs_train, coeffs_me])
    gallery_paths = np.concatenate([
        train_paths,
        np.array([str(p) for p in me_paths])
    ])

    np.save(dst_dir / "coeffs_gallery", coeffs_gallery)
    np.save(dst_dir / "coeffs_me", coeffs_me)
    np.save(dst_dir / "gallery_paths", gallery_paths)
    np.save(dst_dir / "me_paths", np.array([str(p) for p in me_paths]))
    print(f"Done. added coeffs_gallery, coeffs_me, gallery_paths and me_paths in {dst_dir}.")

if __name__ == "__main__":
    project_me_into_eigenbasis()

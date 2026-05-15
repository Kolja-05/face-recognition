import face_recognition
import cv2
from pathlib import Path
from tqdm import tqdm

def process_image(src_path: Path, dst_dir: Path) -> int:
    """
    process image located at path and saves it under the same name in dst_dir

    Arguments
    src_path: path to the image to process
    dst_dir: path of the directory where the processed image should be saved
    """
    src = Path(src_path)
    dst = dst_dir / src.name
    dst = dst.resolve()
    dst = str(dst)

    image = face_recognition.load_image_file(src_path)
    locations = face_recognition.face_locations(image)

    faces_found = len(locations)
    # skip images with more than one face for clean data
    if faces_found > 1:
        return 1
    # skip if now faces were found
    if faces_found < 1:
        return 2
    # whe have exactly 1 face
    (top, right, bottom, left) = locations[0]

    width = right - left
    height = bottom - top

    top = round(top - 0.45 * height)
    bottom = round(bottom + 0.05 * height)
    right = round(right + 0.25 * width)
    left = round(left- 0.25 * width)

    h_img, w_img = image.shape[:2]

    # out of bounds check
    if (top < 0 or left < 0 or bottom > h_img or right > w_img):
        return 3
    if bottom <= top or right <= left:
        return 3

    image = image[:,:,::-1]
    face_subimage = image[top:bottom, left:right]
    if face_subimage is None or face_subimage.size == 0:
        return 3
    face_subimage = cv2.resize(face_subimage, (128,128))

    success = cv2.imwrite(dst, face_subimage)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed to write image {dst}")
    return 0

if __name__ == "__main__":
    src_dir = Path("data/raw")
    dst_dir = Path("data/clean")

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    paths = [p for p in src_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    skipped_more_faces = 0
    skipped_no_face = 0
    skipped_crop_err = 0

    pbar = tqdm(paths, desc="Processing images")

    for path in pbar:
        ret = process_image(path, dst_dir)

        if ret == 1:
            skipped_more_faces += 1
        elif ret == 2:
            skipped_no_face += 1
        elif ret == 3:
            skipped_crop_err += 1

        pbar.set_postfix({
            "multi-face": skipped_more_faces,
            "no-face": skipped_no_face,
            "face to small": skipped_crop_err
        })
            

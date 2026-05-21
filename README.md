# Face-Recognition via Eigenfaces

Real-time face recognition from a webcam, using the [Eigenface-Algorithm](#the-eigenface-algorithm), implemented from scratch in Python.


 # How it works
The system learns a low dimensional space of faces (Eigenspace) from a set of grayscale face images. Each face is represented as vectors of coefficients of the Eigenfaces, which build a basis of the Eigenspace. At recognition time, a query face is projected into the Eigenspace. That projected image is a much more low dimensional vector, which can be used to find a closest match by Euclidean distance.

# Project structure

```
face-recognition/
├── data/
│   ├── clean/              # Raw training face images
│   ├── me/                 # Captured images of the person to recognise
│   └── eigenfaces/         # Generated artefacts (pcs.npy, mean.npy, …)
│
├── capture_me.py           # Step 1 - record your own face via webcam
├── extract_faces_from_raw_data.py  # Step 2 - extract & normalise training faces
├── extract_face_from_webcam.py     # Utility - extract a single face from webcam
├── calculate_eigenfaces.py # Step 3 - run PCA and persist eigenfaces (takes some time)
├── project_me_into_eigenspace.py   # Step 4 - project your face into eigenspace
├── reconstruct_face.py     # Utility - reconstruct a face from its coefficients
└── identify_me.py          # Step 5 - live recognition
```

# Instalation


Note: The face-recognition models work best with Python 3.10

```bash
git clone https://github.com/Kolja-05/face-recognition.git
cd face-recognition
pip install numpy opencv-python face_recognition tqdm
pip install git+https://github.com/ageitgey/face_recognition_models
```
> `face_recognition` requires `dlib`. On most systems a pre-built wheel is available via pip; if not, follow the [dlib installation guide](https://github.com/davisking/dlib).


# Get the LFW-Dataset
```bash
mkdir -p data/raw
cd data/raw
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xvzf lfw.tgz
rm lfw.tgz
```

# How to use

**1. Capture training-set images**
```bash
mkdir -p data/clean
python extract_faces_from_raw_data.py
```
 
**2. Record your own face**
```bash
python capture_me.py
```
Sit in front of the webcam — it captures 600 photos at 0.5 s intervals and saves detected faces to `data/me/`. Press `q` to stop early.
 
**3. Compute eigenfaces**
```bash
python calculate_eigenfaces.py
```
Runs PCA on up to 4 000 training images, selects the top $k$ components (80 % variance threshold) and saves `pcs.npy`, `mean.npy`, `coeffs_train.npy`, and visualisations of each eigenface to `data/eigenfaces/`. This may take some time.
 
**4. Project your face**
```bash
python project_me_into_eigenspace.py
```
 
**5. Run live recognition**
```bash
python identify_me.py
```
 
---
 
## Dependencies
 
| Package | Purpose |
|---|---|
| `numpy` | Linear algebra (SVD, matrix ops) |
| `opencv-python` | Image I/O, webcam access, display |
| `face_recognition` | Face detection (dlib HOG/CNN) |
| `tqdm` | Progress bars |
---


















## The Eigenface Algorithm

**Input**: 
- $m$ $n$-dimenstional Datapoints: $m$ faces as imigas with width $w$ and height $h$ ($n=w \cdot h$), in grayscale. We assume that $m\leq n$.
- One $n$-dimensional query-image: Face with width $w$ and height $h$, in grayscale.

Output:
- The index $i\in \{1, \dots, m\}$ of the face image $g$, that best matches $g^*$.

### 1. Compute the mean face image
$$g = \frac{1}{m}\sum_{i=1}^m g_i \in R^n$$

### 2. Subtract the mean
$$x_i = g_i.g\in R^n$$

### 3. Setup Datamatrix

$$D = \begin{bmatrix} ───x_1^T ───\\\ \vdots \\\ ───x_m^T ─── \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 4. Compute the Singular-Value-Decomposition (SVD)

$$X = U \Sigma V^T=U \begin{bmatrix}
\sigma_1 &             \\
         &         \ddots \\
         &        &        \sigma_m
\end{bmatrix}\begin{bmatrix} ───v_1^T ───\\\ \vdots \\\ ───v_m^T ─── \end{bmatrix}$$

- $U \in \mathbb{R}^{m \times m}$ — orthonormal matrix
- $\Sigma \in \mathbb{R}^{m \times m}$ — diagonal matrix with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m \geq 0$
- $V \in \mathbb{R}^{n \times m}$ — columns $v_1, \ldots, v_m$ are the singular vectors (eigenfaces)

### 5. Select first $k$ singular vectors (Eigenfaces)

$$V_k = \begin{bmatrix} 
| & & |\\
v_1 & \cdots & v_k \\
| & & |
\end{bmatrix} \in \mathbb{R}^{n \times k}$$

### 6. Dimensionality reduction (projection)

$$y_i = V_k^T x_i \in \mathbb{R}^k$$

### 7. Project the query image in point eigenbasis

$$y* = V_k^T x_i \in \mathbb{R}^k$$

### 8. Find closest match in original images

$$i = \underset{i \in \{1,\ldots,m\}}{\arg\min} \lVert y^* - y_i\rVert$$

# EigenMatch: SVD-Based Semantic Resume Alignment 🎯

**EigenMatch** is a Python-based predictive talent matching engine that applies linear algebra transformations to align unstructured candidate resumes with job descriptions. Developed for the **UE25MA242A: Mathematical Foundation for AI & Data Science** course at PES University, this project avoids black-box NLP matching libraries by explicitly implementing matrix factorization and spatial geometry from scratch using NumPy.

By mapping applicant data and job requirements into a shared high-dimensional vector space, EigenMatch identifies semantic overlap based on conceptual direction rather than simple keyword string matching.

---

## Core Architecture & Mathematical Pipeline

The system is built on a 5-phase linear algebra workflow:

### 1. Data Ingestion & Preprocessing
* Generates a controlled synthetic dataset of 5,000 resumes and 500 job descriptions across distinct career clusters.
* Explicitly normalizes text by enforcing a lowercase baseline, stripping punctuation via regex, and manually removing grammatical stop words to prevent false orthogonal dimensions in the vector space.

### 2. Vector Space Mapping (TF-IDF)
* Transforms the cleaned text corpus into mathematical sparse matrices ($A_{resumes}$ and $A_{jobs}$) using Term Frequency-Inverse Document Frequency calculations.
* Formula applied: $W_{i,j} = TF_{i,j} \times \log(N/df_j)$

### 3. System Simplification (SVD)
* Applies Singular Value Decomposition ($A = U \Sigma V^T$) using `np.linalg.svd` to factor the dense matrices.
* Truncates the space to the top 50 principal singular values, mathematically eliminating redundant noise and isolating core semantic competencies.

### 4. Orthogonal Proximity Engine
* Computes the geometric angle (Cosine Similarity) between candidate vectors and job vectors in the compressed 50-dimensional space.
* Utilizes normalized dot products and $L^2$ norms: $\frac{A \cdot B}{\Vert{}A\Vert{}_2 \Vert{}B\Vert{}_2}$

### 5. Matrix Analytics & Visualization
* Renders a $10 \times 10$ correlation heatmap using Seaborn.
* Features a custom **Dynamic Stoploss Algorithm** that forces the matrix to search the full 500-job database to guarantee the inclusion of exact nearest-neighbor matches for the selected candidates, entirely eliminating orthogonal (0.00) dead rows.

---

## Tech Stack & Libraries
* **Language:** Python 3.x
* **Core Mathematics:** `NumPy`
* **Data Manipulation:** `Pandas`
* **Vectorization:** `Scikit-Learn` (`TfidfVectorizer`)
* **Visualization:** `Seaborn`, `Matplotlib`
* **Environment:** Google Colab / Jupyter Notebooks

---

## How to Run (Google Colab)

1. Clone this repository or download the `.ipynb` notebook file.
2. Upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Run the cells sequentially:
   * **Phase 0 & 1** will automatically generate the required `synthetic_5000_resumes.csv` and `synthetic_500_jobs.csv` files directly in your Colab session memory.
   * Proceed through the text cleaning, SVD factorization, and matching engine cells.
4. The final visualization cell will render the optimized correlation heatmap.

---

## Team

* **Badrinath S Kini**
* **Akash K Devang**
* **Akshay Bharadwaj**
* **Abhinav J. K.**

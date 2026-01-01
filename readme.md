# Manga Colorization (Batch & Mac Optimized)

This is a custom wrapper for the [Manga Colorization V2](https://github.com/qweasdd/manga-colorization-v2) project. 

It is designed specifically for **Mac users (M-Series chips)** and **bulk chapter processing**. It automatically handles folder structures, ignores Mac system files, and applies a vibrance filter to make colors pop.

## ✨ Key Features
* **Batch Automation:** Recursively scans a `Manga/` folder and mirrors the structure to `output/`. Perfect for coloring full chapters at once.
* **Mac Optimized:** Built-in support for Apple Silicon (MPS/Metal) acceleration.
* **Color Boost:** Automatically applies a +40% saturation boost to fix the "washed out" look of raw AI output.
* **Crash Proof:** Automatically ignores `.DS_Store` and non-image files that usually crash the original script.

---

## 🛠️ Installation

1.  **Clone this repository**
    ```bash
    git clone <your-repo-url>
    cd manga-colorization-by-saheb
    ```

2.  **Install Dependencies**
    ```bash
    # It is recommended to use a virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    pip install numpy torch torchvision opencv-python pillow matplotlib
    ```

3.  **Download Weights (Required)**
    You need the AI models to make this work. Download them and place them in the correct folders:
    
    * **Generator (The Brain):** [Download Here](https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/view?usp=sharing)
        * *Action:* Rename to `generator.pth` and put it inside the `networks/` folder.
        
    * **Denoiser (Optional but Recommended):** [Download Here](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing)
        * *Action:* Put it inside the `denoising/models/` folder.

---

## 🚀 How to Use

1.  **Prepare your Manga**
    Create a folder named `Manga` in the root directory. Inside it, create subfolders for each chapter.
    
    Structure:
    ```text
    Project/
    ├── Manga/
    │   ├── Chapter 1/
    │   │   ├── 01.jpg
    │   │   └── 02.jpg
    │   └── Chapter 2/
    ├── networks/
    │   └── generator.pth
    └── my_runner.py
    ```

2.  **Run the Script**
    ```bash
    python my_runner.py
    ```

3.  **Get Results**
    The script will automatically create an `output/` folder with the exact same structure as your input, containing your colorized and vibrant pages.

---

## 🤝 Credits & Acknowledgements

This project is a wrapper built around the incredible research and code by **qweasdd**. All core AI inference logic belongs to the original repository.

* **Original Repository:** [qweasdd/manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)
* **Original Demo:** [mangacol.com](https://mangacol.com)

If you use this for research, please cite the original paper/repo.
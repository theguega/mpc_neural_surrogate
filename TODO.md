### 📝 Notebook (Colab) - **@theo**

The goal is to set up the complete training and evaluation pipeline in a Google Colab notebook.

* **Data Preparation**
    * [ ] Import libraries and create **dataloader**
    * [ ] Implement **simple data augmentation**.
    * [ ] Visualization of a random episode
* **Model & Training Setup**
    * [ ] Write the code for the **3 different models**.
    * [ ] Implement the **training loop**.
    * [ ] Implement the **testing loop**.
* **Loss & Optimization**
    * [ ] Implement the **2 loss functions**:
        * [ ] **Mean Squared Error (MSE)**
        * [ ] **Mean Absolute Error (MAE)**
    * [ ] Implement the **optimizer**:
        * [ ] **AdamW**
    * [ ] Implement the **scheduler**:
        * [ ] **ReduceLROnPlateau**
* **Exporting Results**
    * [ ] **Export all results** in a good text file format:
        * [ ] `model_type`, `hyperparameters`
        * [ ] `training loss`, `testing loss`, `type of loss`, `epoch`, `gradient norm`
    * [ ] **Export the model weights** as a `.pth` file.

---

### 📊 Simple Data Visualization File (Python) - **@chatgpt**

Focus on reading the exported results and generating all necessary graphs for the report.

* [ ] Read the data exported from the notebook.
* [ ] Generate a **graph for the report** based on the **offline evaluation** results.
* [ ] Generate a **graph for the report** based on the **online evaluation** results.

---

### ⚙️ Testing Closed Loop File (Python) - **@dexter**

Implement the closed-loop control simulation and comprehensive performance logging.

* **Simulation & Control**
    * [ ] Implement the loop to run **multiple episodes**.
    * [ ] For each episode:
        * [ ] Start from **0 angles**.
        * [ ] Generate a **new target**.
        * [ ] Implement the control logic using **either**:
            * [ ] **MPC Controller**
            * [ ] Load a neural network and use it from the specified **path**.
        * [ ] Step the simulation (iteration).
* **Exporting Results**
    * [ ] **Export all results** in a good text file format:
        * [ ] `controller`: `mpc` or `nn`
        * [ ] `accuracy per episode`
        * [ ] `computation time per iteration`
        * [ ] `computation time per episode`
        * [ ] `computation cost per iteration`
        * [ ] `computation cost per episode` (CPU and memory usage)

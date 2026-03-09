1. Installation

1.1 Install Conda and create a Conda environment

   ```bash
   conda create --name neuoct python=3.13
   conda activate neuoct
   ```

1.2 Clone this repository and Install the required packages.

   ```bash
   git clone https://github.com/Timekisser/NeuOct
   conda install ocnn thsolver tqdm tensorboard opencv-python torchmetrics
   ```

1.3 Install PyTorch=2.6.0 with conda according to the official documentation.

   ```
   pip3 install torch torchvision torchaudio
   ```


2. Training

2.1 Dataset (To be released soon)

2.2 Run the training script

   ```
   python train.py --config vis_shapenet.yaml
   ```

2.3 Render the results

   ```
   python train.py --config vis_shapenet.yaml SOLVER.gpu 0,  \
          SOLVER.run render
   ```
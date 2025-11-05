# SirSLab LeRobot SO-101 Tutorial

## Introduction

The SO-10xARM is a fully open-source robotic arm project launched by TheRobotStudio. It comprises both the leader and follower robotic arms and provides detailed 3D-printing files alongside assembly and operation guides. LeRobot delivers PyTorch-based models, datasets, and tools for real-world robotics, with the goal of reducing the barrier to entry so that anyone can contribute datasets and pretrained models. The framework focuses on imitation learning and ships with pre-trained models, human-gathered datasets, and simulation environments so that you can get started without first assembling a robot. In the coming weeks we plan to expand support for real-world robotics on the most affordable and capable robots currently available.

## Projects Introduction

The SO-ARM10x together with the reComputer Jetson AI robot kit combines high-precision robotic arm control with an AI computing platform, providing a comprehensive development solution. Based on Jetson Orin or AGX Orin hardware and the LeRobot AI framework, this kit can be used for education, research, and industrial automation. This wiki provides the assembly and debugging tutorial for the SO ARM10x and walks through data collection and policy training with LeRobot.

### Get One Now 🖱️

#### Main Features

- **Open-source and low-cost:** An open-source, low-cost robotic arm solution from TheRobotStudio.
- **Integration with LeRobot:** Designed for seamless use with the LeRobot platform.
- **Abundant learning resources:** Assembly and calibration guides plus tutorials for testing, data collection, training, and deployment help you get started quickly.
- **Compatible with NVIDIA:** Deploy the arm kit alongside the reComputer Mini J4012 Orin NX 16 GB.
- **Multi-scene application:** Suited for education, scientific research, automated production, and robotics tasks that demand precise operation.

#### What's New

- **Wiring optimization:** Compared to SO-ARM100, SO-ARM101 improves wiring reliability and avoids disconnections at joint 3. The new wiring also removes prior joint motion limits.
- **Updated gear ratios for the leader arm:** Motors with optimized gear ratios improve performance and eliminate the need for external gearboxes.
- **New functionality:** The leader arm can follow the follower arm in real time, enabling humans to intervene and correct actions during upcoming learning-policy releases.

> **Caution**
> Seeed Studio is only responsible for the hardware quality. These tutorials follow the official documentation. If you encounter software or dependency issues that you cannot resolve, consult the FAQ at the end of this tutorial or contact the LeRobot platform or LeRobot Discord channel.

### Specifications

| Type | SO-ARM100 Arm Kit | SO-ARM100 Arm Kit Pro | SO-ARM101 Arm Kit | SO-ARM101 Arm Kit Pro |
| --- | --- | --- | --- | --- |
| Leader arm motors | 12× ST-3215-C001 (7.4 V) with 1:345 ratio | 12× ST-3215-C001 (7.4 V) with 1:345 ratio | 12× ST-3215-C001 (7.4 V) with 1:345 ratio for joint 2 only; 2× ST-3215-C044 (7.4 V) with 1:191 for joints 1 & 3; 3× ST-3215-C046 (7.4 V) with 1:147 for joints 4–6 | Same as SO-ARM101 Arm Kit |
| Follower arm motors | Same as leader arm | Same as leader arm | Same as SO-ARM100 | Same as SO-ARM100 |
| Power supply | 5 V 4 A | 5 V 4 A | 5 V 4 A (Leader), 12 V 2 A (Follower) | 5 V 4 A (Leader), 12 V 2 A (Follower) |
| Angle sensor | 12-bit magnetic encoder | 12-bit magnetic encoder | 12-bit magnetic encoder | 12-bit magnetic encoder |
| Operating temperature | 0 °C–40 °C | 0 °C–40 °C | 0 °C–40 °C | 0 °C–40 °C |
| Communication | UART | UART | UART | UART |
| Control method | PC | PC | PC | PC |

> **Danger**
> If you purchase the Arm Kit version, both power supplies are 5 V. For the Arm Kit Pro version, use the 5 V supply for the leader arm calibration and steps, and the 12 V supply for the follower arm.

### Bill of Materials (BOM)

| Part | Amount | Included |
| --- | --- | --- |
| Servo motors | 12 | ✅ |
| Motor control board | 2 | ✅ |
| USB-C cable (2 pcs) | 1 | ✅ |
| Power supply | 2 | ✅ |
| Table clamp | 4 | ✅ |
| 3D-printed arm parts | 1 | Optional |

### Initial System Environment

**Ubuntu x86 requirements**

- Ubuntu 22.04
- CUDA 12+
- Python 3.10
- Torch 2.6

**Jetson Orin requirements**

- Jetson JetPack 6.2
- Python 3.10
- Torch 2.6

## Table of Contents

A. [3D Printing Guide](#a-3d-printing-guide)

B. [Install LeRobot](#b-install-lerobot)

C. [Configure the motors](#c-configure-the-motors)

D. [Assembly](#d-assembly)

E. [Calibrate](#e-calibrate)

F. [Teleoperate](#f-teleoperate)

G. [Add cameras](#g-add-cameras)

H. [Record the dataset](#h-record-the-dataset)

I. [Visualize the dataset](#i-visualize-the-dataset)

J. [Replay an episode](#j-replay-an-episode)

K. [Train a policy](#k-train-a-policy)

L. [Evaluate your policy](#l-evaluate-your-policy)

---

## A. 3D Printing Guide

> **Caution**
> Following the official update of SO-101, SO-100 will no longer receive updates and its source files have been removed from the official repository. You can still find the SO-100 sources on MakerWorld. Tutorials and installation methods remain compatible for existing SO-100 owners, and SO-101 print files are fully compatible with SO-100 motor installations.

1. **Choose a printer:** The supplied STL files print well on most FDM printers. Tested settings:
   - Material: PLA+
   - Nozzle diameter & precision: 0.4 mm nozzle at 0.2 mm layer height _or_ 0.6 mm nozzle at 0.4 mm layer height
   - Infill density: 15 %
2. **Set up the printer:**
   - Calibrate and level the bed.
   - Clean the print surface and dry it thoroughly.
   - Apply a thin, even layer of glue if recommended for adhesion.
   - Load filament and ensure printer settings match the recommendations above.
   - Enable supports everywhere except slopes greater than 45° relative to horizontal, and avoid supports in horizontal screw holes.
3. **Print the parts:**
   - STL packs for both leader and follower arms are pre-oriented for minimal supports.
   - For 220 mm × 220 mm beds (e.g., Ender), print the `Follower` and `Leader` bundles.
   - For 205 mm × 250 mm beds (e.g., Prusa/Up), print the corresponding `Follower` and `Leader` bundles.

## B. Install LeRobot

1. **Install Miniconda**
   - **Jetson:**
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
     chmod +x Miniconda3-latest-Linux-aarch64.sh
     ./Miniconda3-latest-Linux-aarch64.sh
     source ~/.bashrc
     ```
   - **Ubuntu x86:**
     ```bash
     mkdir -p ~/miniconda3
     cd ~/miniconda3
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
     bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
     rm ~/miniconda3/miniconda.sh
     source ~/miniconda3/bin/activate
     conda init --all
     ```
2. **Create and activate a fresh environment**
   ```bash
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   ```
3. **Clone LeRobot**
   ```bash
   git clone https://github.com/ZhuYaoHui1998/lerobot.git ~/lerobot
   ```
4. **Optional Orbbec Gemini2 depth camera support**
   ```bash
   cd ~/lerobot
   git checkout orbbec
   ```
   If you only use RGB cameras, remain on `main`. To revert after switching branches:
   ```bash
   cd ~/lerobot
   git checkout main
   ```
5. **Install FFmpeg when using Miniconda**
   ```bash
   conda install ffmpeg -c conda-forge
   ```
   - If `libsvtav1` support is missing:
     ```bash
     conda install ffmpeg=7.1.1 -c conda-forge
     ```
   - On Linux you can also build FFmpeg from source with `libsvtav1` support.
6. **Install LeRobot with Feetech motor dependencies**
   ```bash
   cd ~/lerobot
   pip install -e ".[feetech]"
   ```
7. **Jetson-specific OpenCV & NumPy steps** (skip on x86 Ubuntu)
   ```bash
   conda install -y -c conda-forge "opencv>=4.10.0.84"
   conda remove opencv
   pip3 install opencv-python==4.10.0.84
   conda install -y -c conda-forge ffmpeg
   conda uninstall numpy
   pip3 install numpy==1.26.0
   ```
8. **Verify PyTorch CUDA availability**
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
   If `False`, reinstall PyTorch and Torchvision according to the official instructions (Jetson users should follow NVIDIA's guide).

## C. Configure the Motors

> **Danger**
> Due to official code and servo firmware updates, users before June 30, 2025 must update servo firmware to version 3.10 using the Feetech Windows utility. Power on all servos, select the correct port and baudrate (1 000 000), scan, then run **Upgrade → Online Detection → Upgrade Firmware**. If a servo is not recognized after a failed update, connect a detectable servo, start a scan, then hot-swap the unrecognized servo within one second before retrying the upgrade.

The calibration and initialization process is identical for SO-ARM100 and SO-ARM101, but note that gear ratios differ for the first three joints of the SO-ARM101 leader arm.

1. **Label hardware:** Assign one bus servo adapter and six motors to the leader arm, and the other adapter and six motors to the follower arm. Mark each motor with `F1–F6` or `L1–L6` and keep track of the model and gear ratios:

   | Servo model | Gear ratio | Joints |
   | --- | --- | --- |
   | ST-3215-C044 (7.4 V) | 1:191 | L1, L3 |
   | ST-3215-C001 (7.4 V) | 1:345 | L2 |
   | ST-3215-C046 (7.4 V) | 1:147 | L4–L6 |
   | ST-3215-C001 / C018 / C047 | 1:345 | F1–F6 |

2. **Power requirements:** Use the 5 V supply for STS3215 7.4 V motors and 12 V for STS3215 12 V motors. The leader arm always uses 7.4 V motors—do **not** connect it to 12 V.
3. **Find USB ports:**
   ```bash
   python lerobot/scripts/find_motors_bus_port.py
   ```
   On macOS, ports appear as `/dev/tty.usbmodem*`; on Linux they may be `/dev/ttyACM0`, `/dev/ttyACM1`, etc. If needed on Linux:
   ```bash
   sudo chmod 666 /dev/ttyACM0
   sudo chmod 666 /dev/ttyACM1
   ```
4. **Configure each motor ID:** Plug one motor at a time and run:
   ```bash
   python lerobot/scripts/configure_motor.py \
     --port /dev/ttyACM0 \
     --brand feetech \
     --model sts3215 \
     --baudrate 1000000 \
     --ID 1
   ```
   The script sets ID 1 and centers the motor at position 2048. Repeat for IDs 2–6, then repeat for the other arm on its port.

## D. Assembly

The SO-ARM101 dual-arm assembly mirrors SO-ARM100, with added cable clips and different leader-arm gear ratios. After calibrating servos, avoid rotating them before fastening hardware. Verify motor models and gear ratios before assembly. Follow the official assembly guide for detailed imagery and step-by-step instructions.

> **Danger**
> For SO-ARM101 Arm Kit Standard Edition, all power supplies are 5 V. For the Pro Edition, use 5 V for the leader arm and 12 V for the follower arm during every step.

## E. Calibrate

The SO-ARM100 and SO-ARM101 calibration procedures are compatible; SO-ARM100 users can apply SO-ARM101 parameters and code directly.

> **Danger**
> For the SO-ARM101 Arm Kit Standard Edition, all power supplies are 5 V. For the Pro Edition, use 5 V for the leader arm and 12 V for the follower arm during every step.

1. **Prepare the environment:** Connect both power and USB cables to the SO-10x robot before calibration. Delete `~/lerobot/.cache/huggingface/calibration/so101` to redo calibration.
2. **Update port defaults:** Edit `lerobot/lerobot/common/robot_devices/robots/configs.py` and set the correct serial ports:
   ```python
   @RobotConfig.register_subclass("so101")
   @dataclass
   class So101RobotConfig(ManipulatorRobotConfig):
       calibration_dir: str = ".cache/calibration/so101"
       max_relative_target: int | None = None

       leader_arms: dict[str, MotorsBusConfig] = field(
           default_factory=lambda: {
               "main": FeetechMotorsBusConfig(
                   port="/dev/ttyACM0",  # UPDATE
                   motors={
                       "shoulder_pan": [1, "sts3215"],
                       "shoulder_lift": [2, "sts3215"],
                       "elbow_flex": [3, "sts3215"],
                       "wrist_flex": [4, "sts3215"],
                       "wrist_roll": [5, "sts3215"],
                       "gripper": [6, "sts3215"],
                   },
               ),
           }
       )

       follower_arms: dict[str, MotorsBusConfig] = field(
           default_factory=lambda: {
               "main": FeetechMotorsBusConfig(
                   port="/dev/ttyACM1",  # UPDATE
                   motors={
                       "shoulder_pan": [1, "sts3215"],
                       "shoulder_lift": [2, "sts3215"],
                       "elbow_flex": [3, "sts3215"],
                       "wrist_flex": [4, "sts3215"],
                       "wrist_roll": [5, "sts3215"],
                       "gripper": [6, "sts3215"],
                   },
               ),
           }
       )
   ```
3. **Dual-arm teleoperation (optional):**
   ```bash
   sudo chmod 666 /dev/ttyACM*
   ```
4. **Manual calibration commands:**
   ```bash
   python lerobot/scripts/control_robot.py \
     --robot.type=so101 \
     --robot.cameras='{}' \
     --control.type=calibrate \
     --control.arms='["main_follower"]'

   python lerobot/scripts/control_robot.py \
     --robot.type=so101 \
     --robot.cameras='{}' \
     --control.type=calibrate \
     --control.arms='["main_leader"]'
   ```

## F. Teleoperate

Run a simple teleoperation session without camera streams:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

## G. Add Cameras

1. **Identify camera indices:** Plug cameras directly into the host (avoid USB hubs) and run:
   ```bash
   python lerobot/common/robot_devices/cameras/opencv.py \
       --images-dir outputs/images_from_opencv_cameras
   ```
   The output lists detected indices (e.g., `2`, `4`) and saves sample frames under `outputs/images_from_opencv_cameras`.
2. **Update camera configuration:** Edit `lerobot/lerobot/common/robot_devices/robots/configs.py` to match the discovered indices:
   ```python
   cameras: dict[str, CameraConfig] = field(
       default_factory=lambda: {
           "laptop": OpenCVCameraConfig(
               camera_index=0,  # UPDATE
               fps=30,
               width=640,
               height=480,
           ),
           "phone": OpenCVCameraConfig(
               camera_index=1,  # UPDATE
               fps=30,
               width=640,
               height=480,
           ),
       }
   )
   ```
3. **Optional: Single Orbbec Gemini 2 depth camera:** Configure according to the Orbbec branch instructions.

4. **Display cameras during teleoperation:**
   ```bash
   python lerobot/scripts/control_robot.py \
     --robot.type=so101 \
     --control.type=teleoperate \
     --control.display_data=true
   ```

## H. Record the Dataset

1. **Authenticate with Hugging Face:**
   ```bash
   python - <<'PY'
   from huggingface_hub import login
   import os

   tok = os.environ.get("HUGGINGFACE_TOKEN") or input("Paste your HF token: ")
   login(token=tok, add_to_git_credential=True)
   print("✅ Logged in and git-credential updated.")
   PY
   ```
2. **Record demonstrations:**
   ```bash
   TASK="Grasp the cube and put it in the blue box."
   RUN_ID="so101_grasp_v1_easy_posA_$(date +%Y%m%d_%H%M%S)_01"

   python lerobot/scripts/control_robot.py \
     --robot.type=so101 \
     --control.type=record \
     --control.fps=30 \
     --control.single_task="$TASK" \
     --control.repo_id=seeed_123/${RUN_ID} \
     --control.tags='["grasp","easy","posA","train"]' \
     --control.warmup_time_s=5 \
     --control.episode_time_s=20 \
     --control.reset_time_s=5 \
     --control.num_episodes=50 \
     --control.push_to_hub=false
   ```
   This setup records 50 episodes with 5 s warmup, 20 s episode duration, and 5 s reset intervals. Disabling `--control.push_to_hub` keeps data local until you are ready to upload.

## I. Visualize the Dataset

Use LeRobot's built-in visualization tools or standard notebook workflows to inspect recorded episodes before uploading.

## J. Replay an Episode

Re-run recorded demonstrations through `control_robot.py` with `--control.type=replay` to validate trajectories and camera coverage.

## K. Train a Policy

1. **Upload recorded runs to a single dataset repository:**
   ```bash
   python - <<'PY'
   import os, glob
   from huggingface_hub import HfApi, upload_folder

   api = HfApi()
   HF_REPO = "Angry-Lasagna/so101_grasp_v1_easy_posA"  # unified dataset repo
   api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)

   base = os.path.expanduser("~/.cache/huggingface/lerobot/seeed_123")
   runs = sorted(glob.glob(os.path.join(base, "so101_grasp_v1_easy_posA_*")))
   for r in runs:
       if os.path.isdir(r):
           print("Uploading:", r)
           upload_folder(
               repo_id=HF_REPO,
               repo_type="dataset",
               folder_path=r,
               path_in_repo=os.path.basename(r),
           )
   print("✅ Upload completo su", HF_REPO)
   PY
   ```
2. **Train an ACT policy:**
   ```bash
   python lerobot/scripts/train.py \
     --dataset.repo_id="Angry-Lasagna/so101_dice_circle_v1" \
     --policy.type=act \
     --policy.device=cuda \
     --output_dir="outputs/train/act_dice_circle_v1" \
     --job_name="act_dice_circle_v1" \
     --steps=50000 \
     --batch_size=8 \
     --num_workers=4 \
     --log_freq=50 \
     --save_checkpoint=true \
     --save_freq=5000 \
     --wandb.enable=false
   ```
   Monitor logs under `outputs/train/act_dice_circle_v1` and adjust dataset IDs, step counts, or hardware settings as required.

## L. Evaluate Your Policy

After training, evaluate your policy on held-out episodes or through real-robot rollouts to confirm performance before deployment.

---

By following the sections above you can assemble the SO-101 platform, configure the control stack, record new demonstrations, merge them into a single Hugging Face dataset repository, and train an ACT policy using LeRobot.

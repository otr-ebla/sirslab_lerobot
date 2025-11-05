# SirSLab LeRobot SO-101 Tutorial

This guide walks through collecting demonstrations with the SO-101 robot using LeRobot, uploading the recorded data to a single Hugging Face dataset repository, and training an ACT policy on the resulting dataset. Follow the steps in order to reproduce the workflow that was validated in our lab.

## 1. Authenticate with Hugging Face

Log into the Hugging Face Hub so that LeRobot and `huggingface_hub` can push datasets on your behalf. The snippet below reads your personal access token from the `HUGGINGFACE_TOKEN` environment variable, or prompts you to paste it interactively.

```bash
python - <<'PY'
from huggingface_hub import login
import os

tok = os.environ.get("HUGGINGFACE_TOKEN") or input("Paste your HF token: ")
login(token=tok, add_to_git_credential=True)
print("✅ Logged in and git-credential updated.")
PY
```

## 2. Record demonstrations with LeRobot

Define the task description and a unique run identifier, then launch the LeRobot recorder. Adjust the parameters (episode counts, timing, etc.) to match your experimental setup.

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

This command records 50 episodes, each with a 5 second warmup, 20 second recording window, and 5 second reset interval. Setting `--control.push_to_hub=false` keeps the data in the local Hugging Face cache so you can curate runs before uploading.

## 3. Merge runs into a single dataset repository

After you are satisfied with the recorded runs, upload them to a single Hugging Face dataset repository. The script below scans the LeRobot cache for runs that match the naming convention and uploads each folder into the dataset repository `Angry-Lasagna/so101_grasp_v1_easy_posA`.

```bash
python - <<'PY'
import os, glob
from huggingface_hub import HfApi, upload_folder

api = HfApi()
HF_REPO = "Angry-Lasagna/so101_grasp_v1_easy_posA"  # unico repo dataset
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

The script keeps all runs under the same `repo_id`, resulting in a clean history for the dataset on the Hub.

## 4. Train an ACT policy on the dataset

Once the dataset is available on the Hugging Face Hub, launch training with the ACT policy implementation. The command below is the configuration that produced successful results in our experiments.

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

Monitor training logs in the `outputs/train/act_dice_circle_v1` directory. Adjust the arguments (e.g., `--dataset.repo_id`, `--steps`, or `--policy.device`) as needed for different datasets or hardware configurations.

---

By following these steps you can reproducibly collect new SO-101 demonstrations, consolidate them in a single Hugging Face dataset repository, and train an ACT policy using LeRobot.

import os

ckpt_paths = [
    (
        "weights/kosmos_ph_calvin_abc.pt",
        "weights/kosmos_ph_calvin_abc.json",
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/run_eval_raw_ddp_torchrun.sh {} {}".format(ckpt, config))

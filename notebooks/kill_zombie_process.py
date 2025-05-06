path="/home/lmh/projects_dir/MulConf0420/notebooks/zombie_process.txt"
#每一行形如lmh      1872562  5.4  0.3 129925468 3977460 pts/10 Sl 02:08   0:55 /home/lmh/anaconda3/envs/enzpyg39L/bin/python /home/lmh/projects_dir/MulConf0420/src/train.py trainer=ddp trainer.devices=[0,1,2,3] mode=6 data.batch_size=32 trainer.max_epochs=45 tags=["mode6 ; wt_af3_ref_pdb ; cat ;bs32; cov2repr ;single gearnet layer"] hydra.run.dir="/home/lmh/projects_dir/MulConf0420/logs//train_conf/runs/2025-05-03_02-08-05" hydra.job.name=train_ddp_process_3 hydra.output_subdir=null
#提取出pid
import re
import os
with open(path, 'r') as file:
    for line in file:
        # 使用正则表达式提取出数字
        pid = re.search(r'\d+', line)
        if pid:
            print(pid.group())
            os.system(f"kill -9 {pid.group()}")   

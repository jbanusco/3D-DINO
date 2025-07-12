# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
from pathlib import Path

from dinov2.train import train3d
import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for 3DINO", parents=[train3d.get_args_parser()], add_help=False)
    parser.add_argument("--ngpus", default=4, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    #parser.add_argument("--output_dir", default="", type=str)

    # Lucia specific parameters
    parser.add_argument("--partition", default="gpu", type=str, help="Partition where to submit")
    #parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--job-name', default="3DINO", type=str)
    parser.add_argument("--output", default="/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out", type=str)
    parser.add_argument("--cpus-per-task", default=8, type=int)
    parser.add_argument("--mem", default="220G", type=str)
    parser.add_argument("--time", default="1-23:59:58", type=str)
    parser.add_argument("--mail-user", default="benoit.gerin@uclouvain.be", type=str)
    parser.add_argument("--mail-type", default="ALL", type=str)
    parser.add_argument("--account", default="danitim", type=str)

    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/gpfs/home/acad/ucl-elen/gerinb/submitit/").is_dir():
        p = Path(f"/gpfs/home/acad/ucl-elen/gerinb/submitit/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import train3d

        self._setup_gpu_args()
        train3d.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "job"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=2)

    args.config_file = "dinov2/configs/lucia_ssl3d_default_config.yaml"
    args.cache_dir = "/gpfs/projects/acad/danitim/gerinb/cell_profiling/data/FOMO25/cache"


    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    """ Here, cluster specific parameters can be set."""

    #kwargs['slurm_partition'] = args.partition  # Note: you're setting this twice
    kwargs['slurm_job_name'] = args.job_name
    # Remove slurm_output - not a valid parameter
    # kwargs['slurm_output'] = args.output
    kwargs['slurm_mem'] = args.mem
    kwargs['slurm_time'] = args.time
    kwargs['slurm_mail_user'] = args.mail_user
    kwargs['slurm_mail_type'] = args.mail_type
    kwargs['slurm_account'] = args.account


    executor.update_parameters(
        mem_gb=220,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_gpus_per_node=num_gpus_per_node,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        stderr_to_stdout=True,  # Use this instead of output
        **kwargs
    )

    executor.update_parameters(name="3DINO")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
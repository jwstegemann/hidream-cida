import os
import shutil
import pathlib
import threading
import time
import sys


# ----------------------------------------------------------------------
#  Dienstprogramme
# ----------------------------------------------------------------------
def start_monitoring_disk_space(interval: int = 60) -> None:
    def _log() -> None:
        while True:
            statvfs = os.statvfs("/")
            free = statvfs.f_frsize * statvfs.f_bavail / 1024 ** 3
            print(f"[{os.getenv('MODAL_TASK_ID')}] free disk: {free:6.1f} GB", flush=True)
            time.sleep(interval)

    t = threading.Thread(target=_log, daemon=True)
    t.start()


def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_concurrent(src: pathlib.Path, dest: pathlib.Path) -> None:
    """
    A modified shutil.copytree which copies in parallel to increase bandwidth
    and compensate for the increased IO latency of volume mounts.
    """
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)
            self.copy_jobs = []

        def copy(self, source, dest):
            res = self.pool.apply_async(
                shutil.copyfile,
                args=(source, dest),
                callback=lambda r: print(f"{source} copied to {dest}"),
                # NOTE: this should `raise` an exception for proper reliability.
                error_callback=lambda exc: print(
                    f"{source} failed: {exc}", file=sys.stderr
                ),
            )
            self.copy_jobs.append(res)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    with MultithreadedCopier(max_threads=24) as copier:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True)

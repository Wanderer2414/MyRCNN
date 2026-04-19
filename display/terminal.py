import sys
import time
import shutil
import math
def show_progress(
    current: int,
    total: int,
    start_time: float,
    message: str = "Processing",
    size_label: str = "",
    bar_width: int = 120
) -> None:
    """
    Display a progress bar in the format:
    [00:00:00] Message (size) ██████████..... 100%

    Parameters:
        current (int): current progress value
        total (int): total value
        start_time (float): time.time() when processing started
        message (str): text message
        size_label (str): e.g. '12 Mo'
        bar_width (int): width of the progress bar
    """

    # Elapsed time
    elapsed = int(time.time() - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    time_str = f"[{h:02d}:{m:02d}:{s:02d}]"

    # Progress
    percent = min(current / total, 1.0)
    filled = int(bar_width * percent)
    bar = "█" * filled + " " * (bar_width - filled)

    percent_str = f"{percent * 100:6.2f}%"

    # Line
    line = (
        f"{time_str} {message}"
        f"{f' ({size_label})' if size_label else ''}   "
        f"{bar} {percent_str}"
    )

    # Print in-place
    sys.stdout.write("\r" + line)
    sys.stdout.flush()

    if current >= total:
        print()  # newline at end

def show_progress_counter(
    current: int,
    total: int,
    start_time: float,
    message: str
) -> None:
    """
    Display progress filling the terminal width:
    [HH:MM:SS] Message ███████████ current / total
    """

    term_width = shutil.get_terminal_size(fallback=(120, 20)).columns
    elapsed = math.ceil((time.time() - start_time)/current*(total - current)) if (current < total) else math.ceil(time.time()-start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    time_str = f"[{h:02d}:{m:02d}:{s:02d}]"
    counter = f"{current:<9}/{total:9}"
    prefix = f"{time_str} {message:<30} "
    suffix = f" {counter}"

    static_len = len(prefix) + len(suffix)
    bar_width = max(10, term_width - static_len)

    ratio = min(current / total, 1.0)
    filled = int(bar_width * ratio)
    bar = "█" * filled + " " * (bar_width - filled)

    line = prefix + bar + suffix

    sys.stdout.write("\r" + line[:term_width])
    sys.stdout.flush()

    if current >= total:
        print()

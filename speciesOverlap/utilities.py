import sys

import numpy as np

def update_progress(progress_id, N_id, times, bar_length=40, status=""):        
    """
    This function was coded by Marc Wiedermann (https://github.com/marcwie)

    Draw a progressbar according to the actual progress of a calculation.

    Call this function again to update the progressbar.

    :type progress: number (float)
    :arg  progress: Percentile of progress of a current calculation.

    :type bar_length: number (int)
    :arg  bar_length: The length of the bar in the commandline.

    :type status: str
    :arg  status: A message to print behing the progressbar.
    """
    progress = float(progress_id) / N_id
    block = int(round(bar_length*progress))
    if progress==1.:
        timeleft = "done"
    elif len(times)>20:
        timeleft = int((N_id-progress_id) * np.mean(times))
        timeleft = " - estimated time remaining: " +  _get_timeleft_string(timeleft)
    else:
        timeleft = "... get time estimation ..."
    text = "\r[{0}] {1}% {2}".format("="*block + " "*(bar_length-block),
                                     round(progress, 3)*100, status+" "+timeleft)
    sys.stdout.write(text)
    sys.stdout.flush()
    if progress >= 1:
        sys.stdout.write("\n")

def _get_timeleft_string(t):
    d, remainder = divmod(t,24*60*60)
    h, remainder = divmod(remainder,60*60)
    m, s = divmod(remainder,60)
    t = [d,h,m,s]
    t_str = ["%dd", "%dh", "%dm", "%ds"]
    it = 0 
    while it<4 and t[it]==0.:
        it += 1
    text = (" ".join(t_str[it:it+2])) % tuple(t[it:it+2])
    return text

def _get_sizeof_string(t):
    GB, remainder = divmod(t,1024**3)
    MB, remainder = divmod(remainder,1024**2)
    KB, B = divmod(remainder,1024)
    t_float = t/1024**np.arange(3,-1,-1)
    t = [GB,MB,KB,B]
    t_str = ["%.1fGB", "%.dMB", "%dKB", "%dB"]
    it = 0 
    while it<4 and t[it]==0.:
        it += 1
    text = t_str[it] % (t[it])
    return text


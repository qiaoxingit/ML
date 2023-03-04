import datetime
import time

def log(message):
    current_time = time.time()
    current_time_str = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
    return f'{current_time_str}: {message}'
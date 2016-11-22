import threading
import time
try:
    import gevent
    import gevent.monkey
    from gevent.pool import Pool
except:
    pass

def multi_thread(func,args_list,max_num = 10,pause = 0.001):
    
    max_num = threading.active_count() + max_num
    threads = []
    loc = 0
    hlen = len(args_list)
    while loc < hlen:
        time.sleep(pause)
        if threading.active_count() < max_num:
            t = threading.Thread(target = func, args = args_list[loc])
            t.setDaemon(True)
            t.start()
            threads.append(t)
            loc += 1
    completed = 0
    
    while completed < hlen:
        time.sleep(pause)
        completed = 0
        for t in threads:
            if not t.is_alive():
                completed += 1
                
    return threads
    
def multi_thread_gevent(func,args_list,max_num = 10):
    gevent.monkey.patch_socket()
    pool = Pool(max_num)
    pool.map(func,args_list)
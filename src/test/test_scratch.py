def test1():
    """
    test whether queues can be passed into torch.mp subprocesses in arbitrary ways
    """

    # Stackoverflow example https://stackoverflow.com/questions/11515944/how-to-use-multiprocessing-queue-in-python
    from torch import multiprocessing as mp
    from queue import Queue
    import time

    def reader(kwargs):
        ## Read from the queue
        queue = kwargs["hello"]
        while True:
            msg = queue.get()         # Read from the queue and do nothing
            if (msg == 'DONE'):
                break

    def writer(count, queue):
        ## Write to the queue
        for ii in range(0, count):
            queue.put(ii)             # Write 'count' numbers into the queue
        queue.put('DONE')

    if __name__=='__main__':
        for count in [10**4, 10**5, 10**6]:
            queue = [mp.Queue()]   # reader() reads from queue
                              # writer() writes to queue
            reader_p = mp.Process(target=reader, args=(dict(hello=queue[0]),))
            reader_p.daemon = True
            reader_p.start()        # Launch reader() as a separate python process

            _start = time.time()
            writer(count, queue[0])    # Send a lot of stuff to reader()
            reader_p.join()         # Wait for the reader to finish
            print("Sending %s numbers to Queue() took %s seconds" % (count,
                (time.time() - _start)))

def main():
    test1()
    pass

if __name__ == "__main__":
    main()
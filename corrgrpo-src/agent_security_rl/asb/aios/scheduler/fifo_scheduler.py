# This implements a (mostly) FIFO task queue using threads and queue, in a
# similar fashion to the round robin scheduler. However, the timeout is 1 second
# instead of 0.05 seconds.

from .base import BaseScheduler

from queue import Queue, Empty
from threading import Thread

import os
import time

from pyopenagi.queues.llm_request_queue import LLMRequestQueue

class FIFOScheduler(BaseScheduler):
    def __init__(self, llm, log_mode):
        super().__init__(llm, log_mode)
        self.agent_process_queue = Queue()
        self.worker_count = max(1, int(os.environ.get("ASB_SCHEDULER_WORKERS", "1")))
        self.threads = [self.thread] + [
            Thread(target=self.run, name=f"asb-llm-worker-{index}")
            for index in range(1, self.worker_count)
        ]

    def start(self):
        self.active = True
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.active = False
        for thread in self.threads:
            thread.join()


    def run(self):
        while self.active:
            try:
                """
                wait 1 second between each iteration at the minimum
                if there is nothing received in a second, it will raise Empty
                """
                # agent_process = self.agent_process_queue.get(block=True, timeout=1)
                agent_process = LLMRequestQueue.get_message()
                # print("Get the request")
                agent_process.set_status("executing")
                self.logger.log(f"{agent_process.agent_name} is executing. \n", "execute")
                agent_process.set_start_time(time.time())
                self.execute_request(agent_process)
            except Empty:
                pass

    def execute_request(self, agent_process):
        self.llm.address_request(
            agent_process=agent_process
        )

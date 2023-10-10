import asyncio
import threading

class RateLimiter:
    def __init__(self, rate: int, period: int, retries: int = 3):
        self.rate = rate
        self.period = period
        self.retries = retries
        self.semaphore = asyncio.Semaphore(rate)
        self.tasks = []

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.tasks.append(asyncio.create_task(self.release()))

    async def release(self):
        await asyncio.sleep(self.period)
        self.semaphore.release()
        self.tasks.remove(asyncio.current_task())

    async def execute(self, task, *args, **kwargs):
        for _ in range(self.retries):
            try:
                async with self:
                    return await task(*args, **kwargs)
            except Exception as e:
                if _ < self.retries - 1:  # If not the last retry
                    await asyncio.sleep(self.period)
                else:
                    raise e from None
                
class SyncRateLimiter:
    def __init__(self, rate: int, period: int, max_retries: int = 3):
        self.rate = rate
        self.period = period
        self.max_retries = max_retries
        self.semaphore = threading.Semaphore(rate)
        self.lock = threading.Lock()
        self.timer = None

    def _release_semaphore(self):
        with self.lock:
            self.semaphore.release()

    def _schedule_release(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.period, self._release_semaphore)
        self.timer.start()

    def execute(self, func, *args, **kwargs):
        for _ in range(self.max_retries):
            acquired = self.semaphore.acquire(blocking=False)
            if acquired:
                try:
                    self._schedule_release()
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Exception occurred: {e}. Retrying...")
                    self.semaphore.release()
            else:
                threading.Event().wait(self.period)

        raise Exception("Max retries reached")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
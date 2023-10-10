import asyncio

class RateLimiter:
    def __init__(self, rate: int, period: int):
        self.rate = rate
        self.period = period
        self.semaphore = asyncio.Semaphore(rate)
        self.tasks = []

    async def __aenter__(self):
        await self.semaphore.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.tasks.append(asyncio.create_task(self.release()))

    async def release(self):
        await asyncio.sleep(self.period)
        self.semaphore.release()
        self.tasks.remove(asyncio.current_task())

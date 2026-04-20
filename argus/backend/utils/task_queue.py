"""
ARGUS-X — Bounded Background Task Queue

Provides controlled async task execution with:
- Fixed-size queue (backpressure via rejection when full)
- Fixed worker count (concurrency control)
- Exception logging (no silent failures)
- Graceful shutdown
- Observable status for /health endpoint

Replaces all unbounded asyncio.create_task() calls in the system.
"""
import asyncio
import logging
from typing import Callable, Awaitable

log = logging.getLogger("argus.task_queue")


class BackgroundTaskQueue:
    """
    Bounded async work queue with a fixed worker pool.

    Prevents unbounded task accumulation under load.
    When the queue is full, new tasks are DROPPED (logged, not silent).
    This is deliberate backpressure — better to skip a mutation than crash.
    """

    def __init__(self, maxsize: int = 50, workers: int = 3, name: str = "bg"):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._workers = workers
        self._name = name
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._processed = 0
        self._dropped = 0

    async def start(self):
        """Start worker pool. Call during app lifespan startup."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._worker(i))
            for i in range(self._workers)
        ]
        log.info(
            f"✅ TaskQueue[{self._name}] started: "
            f"{self._workers} workers, max {self._queue.maxsize} queued"
        )

    async def _worker(self, worker_id: int):
        """Worker loop: pulls tasks from queue and executes them."""
        while self._running:
            try:
                coro_factory, label = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue  # Check self._running flag
            except asyncio.CancelledError:
                break

            try:
                await coro_factory()
                self._processed += 1
            except Exception as e:
                log.warning(f"[{self._name}-W{worker_id}] {label} failed: {e}")
            finally:
                self._queue.task_done()

    def enqueue(
        self, coro_factory: Callable[[], Awaitable], label: str = "task"
    ) -> bool:
        """
        Enqueue a task. Returns True if accepted, False if queue is full.

        IMPORTANT: coro_factory must be a zero-arg callable that RETURNS a
        coroutine (not an already-created coroutine). This prevents
        'coroutine was never awaited' warnings when tasks are dropped.

        Usage:
            queue.enqueue(lambda: some_async_fn(arg1, arg2), "my-task")
        """
        try:
            self._queue.put_nowait((coro_factory, label))
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            log.warning(
                f"[{self._name}] Queue full ({self._queue.maxsize}) — "
                f"dropping '{label}' (total dropped: {self._dropped})"
            )
            return False

    def status(self) -> dict:
        """Observable status for /health endpoint."""
        return {
            "name": self._name,
            "queued": self._queue.qsize(),
            "max_size": self._queue.maxsize,
            "workers": self._workers,
            "processed": self._processed,
            "dropped": self._dropped,
            "running": self._running,
        }

    async def shutdown(self):
        """Graceful shutdown: stop accepting, finish current work, cancel workers."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info(
            f"🔴 TaskQueue[{self._name}] shutdown — "
            f"processed: {self._processed}, dropped: {self._dropped}"
        )

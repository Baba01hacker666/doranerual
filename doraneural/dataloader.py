"""Thread-based batch-parallel data loader with prefetching.

Zero heavy dependencies (uses standard library threading & queue).
Overlaps batch preparation/augmentation on background threads with CPU compute in the main thread.
"""

import queue
import threading
from typing import Iterator, Tuple, Optional, Callable, Any, Union, Sequence
import numpy as np


class Dataset:
    """Abstract base dataset representing a mapped sequence of data samples."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class ArrayDataset(Dataset):
    """In-memory dataset wrapping feature and label arrays."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X: np.ndarray = np.asarray(X)
        self.y: Optional[np.ndarray] = np.asarray(y) if y is not None else None
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(
                f"Sample count mismatch: X has {len(self.X)} samples, y has {len(self.y)} samples."
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]


class _Sentinel:
    """Unique sentinel token marking end of iteration stream."""
    pass


class DataLoader:
    """Thread-based Batch-Parallel DataLoader with background prefetching.

    Overlaps data slicing, memory formatting, and augmentation on a background thread
    while the main thread executes forward and backward passes.

    Args:
        dataset (Union[Dataset, Tuple[np.ndarray, np.ndarray]]): Dataset instance or (X, y) array tuple.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle sample order at the beginning of each epoch.
        prefetch_factor (int): Number of batches precomputed and buffered in queue.
        drop_last (bool): Whether to drop the trailing incomplete batch if dataset size is not divisible by batch_size.
        collate_fn (Optional[Callable]): Custom batch collator function.
        transform (Optional[Callable]): Optional function applied to each (X_batch, y_batch) before yielding.
        seed (Optional[int]): Random seed for reproducible batch shuffling.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32,
        shuffle: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(dataset, tuple) and len(dataset) == 2:
            self.dataset: Dataset = ArrayDataset(dataset[0], dataset[1])
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError(
                f"Expected Dataset instance or (X, y) tuple, got {type(dataset)}"
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

        self.batch_size: int = int(batch_size)
        self.shuffle: bool = bool(shuffle)
        self.prefetch_factor: int = int(prefetch_factor)
        self.drop_last: bool = bool(drop_last)
        self.collate_fn: Optional[Callable] = collate_fn
        self.transform: Optional[Callable] = transform
        self.seed: Optional[int] = seed

        self._queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _worker_fn(self, q: queue.Queue, stop_event: threading.Event, indices: np.ndarray) -> None:
        """Background worker thread that slices and pushes precomputed batches to queue."""
        num_samples = len(indices)
        step = self.batch_size

        try:
            for start_idx in range(0, num_samples, step):
                if stop_event.is_set():
                    break

                batch_indices = indices[start_idx : start_idx + step]
                if self.drop_last and len(batch_indices) < self.batch_size:
                    break

                # Slice batch
                batch_data = self.dataset[batch_indices]

                # Optional user transform / augmentation
                if self.transform is not None:
                    if isinstance(batch_data, tuple):
                        batch_data = self.transform(*batch_data)
                    else:
                        batch_data = self.transform(batch_data)

                # Optional custom collate
                if self.collate_fn is not None:
                    batch_data = self.collate_fn(batch_data)

                # Put into bounded prefetch queue (blocks if queue is full until consumer dequeues)
                while not stop_event.is_set():
                    try:
                        q.put(batch_data, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        except Exception as exc:
            q.put(exc)
        finally:
            # Signal iteration completion
            q.put(_Sentinel())

    def __iter__(self) -> Iterator[Any]:
        # Clean up any leftover thread from previous partial loop
        self.close()

        num_samples = len(self.dataset)
        if self.shuffle:
            indices = self._rng.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        self._queue = queue.Queue(maxsize=self.prefetch_factor)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_fn,
            args=(self._queue, self._stop_event, indices),
            daemon=True,
        )
        self._worker_thread.start()

        return self

    def __next__(self) -> Any:
        if self._queue is None or self._stop_event is None:
            raise StopIteration

        item = self._queue.get()
        if isinstance(item, _Sentinel):
            self.close()
            raise StopIteration
        if isinstance(item, Exception):
            self.close()
            raise item

        return item

    def close(self) -> None:
        """Terminate background worker thread and drain prefetch queue."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._queue is not None:
            # Drain queue to prevent worker from blocking on put
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        self._queue = None
        self._worker_thread = None
        self._stop_event = None

    def __enter__(self) -> "DataLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

# SPDX-License-Identifier: MIT
"""
Custom Celery exceptions and task base classes used by gpuphot_worker.

This module provides:
- SerializableTaskError: an exception type that can be converted to a dict
  for storing structured error metadata in Celery task states.
- BaseTaskWithFailureHandling: a Celery Task base class that intercepts
  failures and stores serialized error metadata when SerializableTaskError
  is raised.

All docstrings are written in English and follow NumPy-style where useful.
"""

from celery import Task
from celery.exceptions import TaskError


class SerializableTaskError(TaskError):
    """
    A custom exception class that can be serialized for Celery tasks.

    This class extends TaskError to provide a serializable format for task errors,
    including the exception type and message.

    Parameters
    ----------
    message : str
        The error message.
    exc_type : str, optional
        The exception type, defaults to the class name if not provided.
    """

    def __init__(self, message, exc_type=None):
        super().__init__(message)
        self.exc_type = exc_type or self.__class__.__name__

    def as_dict(self):
        """
        Convert the exception to a dictionary format suitable for Celery metadata.

        Returns
        -------
        dict
            A dictionary containing the exception type and message.
        """
        return {"exc_type": self.exc_type, "message": str(self)}

    def __str__(self):
        """
        Return a short string representation of the exception.

        Returns
        -------
        str
            A string in the format "exception_type: message".
        """
        return f"{self.exc_type}: {super().__str__()}"


class BaseTaskWithFailureHandling(Task):
    """
    A base Celery task class with custom failure handling.

    This class extends the Celery :class:`Task` class to provide custom handling for
    :class:`SerializableTaskError` exceptions. When such an exception is raised the
    task state is updated with serialized error metadata to make debugging easier
    for asynchronous workers and remote callers.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Handle task failure, with special treatment for :class:`SerializableTaskError`.

        This method updates the task state to ``FAILURE`` and includes the serialized
        error information for :class:`SerializableTaskError` exceptions.

        Parameters
        ----------
        exc : Exception
            The exception that caused the task failure.
        task_id : str
            The ID of the failed task.
        args : tuple
            Positional arguments passed to the task.
        kwargs : dict
            Keyword arguments passed to the task.
        einfo : ExceptionInfo
            Extended information about the exception (traceback, etc.).
        """
        if isinstance(exc, SerializableTaskError):
            self.update_state(
                state="FAILURE",
                meta=exc.as_dict(),
            )
        super().on_failure(exc, task_id, args, kwargs, einfo)

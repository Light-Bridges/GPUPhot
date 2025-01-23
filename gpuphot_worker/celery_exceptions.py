from celery import Task
from celery.exceptions import TaskError


class SerializableTaskError(TaskError):
    """
    A custom exception class that can be serialized for Celery tasks.

    This class extends TaskError to provide a serializable format for task errors,
    including the exception type and message.

    :param message: The error message.
    :type message: str
    :param exc_type: The exception type, defaults to the class name if not provided.
    :type exc_type: str, optional
    """

    def __init__(self, message, exc_type=None):
        super().__init__(message)
        self.exc_type = exc_type or self.__class__.__name__

    def as_dict(self):
        """
        Convert the exception to a dictionary format.

        :return: A dictionary containing the exception type and message.
        :rtype: dict
        """
        return {"exc_type": self.exc_type, "message": str(self)}

    def __str__(self):
        """
        Return a string representation of the exception.

        :return: A string in the format "exception_type: message".
        :rtype: str
        """
        return f"{self.exc_type}: {super().__str__()}"


class BaseTaskWithFailureHandling(Task):
    """
    A base Celery task class with custom failure handling.

    This class extends the Celery Task class to provide custom handling for
    SerializableTaskError exceptions.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Handle task failure, with special treatment for SerializableTaskError.

        This method updates the task state to "FAILURE" and includes the serialized
        error information for SerializableTaskError exceptions.

        :param exc: The exception that caused the task failure.
        :type exc: Exception
        :param task_id: The ID of the failed task.
        :type task_id: str
        :param args: Positional arguments passed to the task.
        :type args: tuple
        :param kwargs: Keyword arguments passed to the task.
        :type kwargs: dict
        :param einfo: Extended information about the exception.
        :type einfo: ExceptionInfo
        """
        if isinstance(exc, SerializableTaskError):
            self.update_state(
                state="FAILURE",
                meta=exc.as_dict(),
            )
        super().on_failure(exc, task_id, args, kwargs, einfo)

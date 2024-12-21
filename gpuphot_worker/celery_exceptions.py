from celery import Task
from celery.exceptions import TaskError


class SerializableTaskError(TaskError):
    def __init__(self, message, exc_type=None):
        super().__init__(message)
        self.exc_type = exc_type or self.__class__.__name__

    def as_dict(self):
        return {"exc_type": self.exc_type, "message": str(self)}

    def __str__(self):
        return f"{self.exc_type}: {super().__str__()}"


class BaseTaskWithFailureHandling(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if isinstance(exc, SerializableTaskError):
            self.update_state(
                state="FAILURE",
                meta=exc.as_dict(),
            )
        super().on_failure(exc, task_id, args, kwargs, einfo)

"""Domain-level exceptions.

These are deliberately *not* pydantic `ValidationError`s. The pipeline needs to
tell three things apart:

* transport failures (retry / rotate the API key)      -> `ProviderError`
* the model returned unparsable JSON                   -> pydantic `ValidationError`
* the model returned valid JSON that breaks a rule     -> `DomainValidationError`

Only the last two are worth feeding back to the model; the first one is not the
model's fault and must never end up inside a prompt.
"""


class NeuroclipError(Exception):
    """Base class for everything this application raises on purpose."""


class ProviderError(NeuroclipError):
    """Transport / API failure. Never shown to the model, only logged.

    `trail` holds one `model:status` marker per distinct failure - the shortest
    thing that answers "was the key rejected, or does that model not exist?".
    `upstream_reason` carries the provider's own explanation, redacted, for the
    non-retryable case where that explanation is the whole answer.
    """

    trail: list[str] = []
    upstream_reason: str = ""


class ProviderExhaustedError(ProviderError):
    """Every key and every model in the fallback chain has been tried.

    `trail` holds one `model:status` entry per distinct failure - the shortest
    thing that answers "was the key rejected, or does that model not exist?".
    """

    trail: list[str] = []


class DomainValidationError(NeuroclipError):
    """The payload parsed fine but violates a business rule.

    The message is written to be readable by the model: it goes straight into
    the repair prompt.
    """


class RepairExhaustedError(NeuroclipError):
    """The repair loop ran out of attempts."""

    def __init__(self, attempts: int, last_error: str):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to obtain a valid structure after {attempts} attempt(s). "
            f"Last error: {last_error}"
        )


class InfeasibleBriefError(NeuroclipError):
    """The brief cannot be satisfied by the domain rules at all.

    Raised before any model call, e.g. a 15-second video when the renderer only
    produces 4/6/8-second clips.
    """

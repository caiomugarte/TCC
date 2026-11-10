"""Failures the CLI reports as one stderr line and an exit code."""

from datetime import date


class RebuildError(Exception):
    """A failure the CLI reports as one stderr line and an exit code."""

    exit_code = 1

    def __init__(self, field_name: str, reason: str):
        super().__init__(f"{field_name}: {reason}")
        self.field_name = field_name
        self.reason = reason


class ArgumentError(RebuildError):
    exit_code = 2


class MissingPathError(RebuildError):
    exit_code = 3


class IntegrityError(RebuildError):
    exit_code = 4


class PITLeakError(IntegrityError):
    """A value dated after its decision date reached the computation (PIT-02)."""

    def __init__(self, ticker: str, decision_date: date, offending_date: date):
        super().__init__(
            ticker,
            f"decision {decision_date.isoformat()} received a value dated "
            f"{offending_date.isoformat()}",
        )
        self.ticker = ticker
        self.decision_date = decision_date
        self.offending_date = offending_date


class SourceError(RebuildError):
    """A required download failed, or a required file is absent offline."""

    exit_code = 5


class OutputError(RebuildError):
    """The report files could not be written (CLI-01, RPT-05)."""

    exit_code = 1

"""Tools the model may ask to run, and the effect class that gates them (SPEC §13.3)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

logger = logging.getLogger("cairn.tools")

Effect = Literal["read", "write"]


@dataclass(frozen=True)
class Tool:
    """One callable the model may request.

    `effect` defaults to `"write"`, so a tool nobody classified needs approval.
    The two mistakes are not symmetric: a misfiled `read` costs one unnecessary
    prompt, a misfiled `write` sends an email nobody approved.
    """

    name: str
    description: str
    run: Callable[..., Awaitable[str]]
    effect: Effect = "write"
    # What a human may change while approving. Never the target of the effect.
    editable_fields: frozenset[str] = field(default_factory=frozenset)

    @property
    def needs_approval(self) -> bool:
        return self.effect == "write"

    def signature(self) -> str:
        params = ", ".join(inspect.signature(self.run).parameters)
        return f"{self.name}({params})"

    def validate(self, args: dict[str, Any]) -> bool:
        """Do these arguments actually call this tool? Checked before proposing."""
        try:
            inspect.signature(self.run).bind(**args)
        except TypeError:
            return False
        return True

    def preview(self, args: dict[str, Any]) -> str:
        """One line describing the effect, for the human deciding on it."""
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{self.name}({rendered})"


class ToolRegistry:
    """The tools a graph may reach. Empty unless one is built and passed in."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools or []}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


class Transport(Protocol):
    """Where a sent email actually goes. A seam, like `VectorStore`."""

    async def send(self, *, to: str, subject: str, body: str) -> str: ...


class LoggingTransport:
    """Default transport: records the intent, sends nothing.

    Real egress belongs to a deployment, not to this repository -- a default that
    reached the network would make every test run a live send.
    """

    async def send(self, *, to: str, subject: str, body: str) -> str:
        """See `Transport.send`."""
        logger.info("send_email to=%s subject=%s chars=%d", to, subject, len(body))
        return f"Email to {to} with subject {subject!r} was handed to the transport."


def build_registry(transport: Transport | None = None) -> ToolRegistry:
    """The default tool set: one drafting tool, one that leaves the process."""
    sender = transport or LoggingTransport()

    async def draft_email(to: str, subject: str, body: str) -> str:
        return f"Draft for {to}\nSubject: {subject}\n\n{body}"

    async def send_email(to: str, subject: str, body: str) -> str:
        return await sender.send(to=to, subject=subject, body=body)

    return ToolRegistry(
        [
            Tool(
                name="draft_email",
                description="Compose an email and return it without sending.",
                run=draft_email,
                effect="read",
            ),
            Tool(
                name="send_email",
                description="Send an email. Leaves the process; needs approval.",
                run=send_email,
                editable_fields=frozenset({"subject", "body"}),
            ),
        ]
    )

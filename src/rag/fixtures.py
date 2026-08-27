"""Seed corpus for the stubbed vector store.

>>> SEAM <<<
The corpus ingestion / ETL pipeline is explicitly OUT OF SCOPE (SPEC §3, §11:
"confirm the vector store and metadata schema are owned elsewhere"). These
fixture documents exist only so the graph, the API and the tests can run
end-to-end against a real `VectorStore` implementation.

Replacing this with a real index means implementing the `VectorStore` protocol
in src/rag/retrieve.py -- nothing else in the codebase should need to change.
The metadata contract that a real store MUST honour is `source` (citable) and
`score` (comparable, higher is better).
"""

from __future__ import annotations

from typing import NamedTuple


class SeedDoc(NamedTuple):
    source: str
    text: str


SEED_DOCS: tuple[SeedDoc, ...] = (
    SeedDoc(
        source="doc://kb/expenses-1",
        text=(
            "Expense reports must be submitted within 30 days of the purchase date. "
            "Receipts are required for any expense over 25 dollars. "
            "Reports submitted after 60 days require director approval."
        ),
    ),
    SeedDoc(
        source="doc://kb/expenses-2",
        text=(
            "Reimbursement for an approved expense report is paid out in the next "
            "payroll cycle, typically within 14 days of approval."
        ),
    ),
    SeedDoc(
        source="doc://kb/vpn-1",
        text=(
            "To connect to the corporate VPN, install the client, sign in with your "
            "single sign-on account, and approve the multi-factor prompt on your phone. "
            "The VPN is required for access to internal databases."
        ),
    ),
    SeedDoc(
        source="doc://kb/onboarding-1",
        text=(
            "New employees complete onboarding in their first week: hardware pickup on "
            "day one, security training by day three, and a manager check-in on day five."
        ),
    ),
    SeedDoc(
        source="doc://kb/invoicing-1",
        text=(
            "Supplier invoices are processed by accounts payable. An invoice must "
            "reference a valid purchase order number or it is returned to the supplier."
        ),
    ),
    SeedDoc(
        source="doc://kb/support-1",
        text=(
            "The support team answers priority one tickets within one hour and priority "
            "three tickets within two business days."
        ),
    ),
)

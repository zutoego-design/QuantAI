# Data Limitations

## Free-Tier Constraints

The 2010+ Nasdaq history is an approximation built from monthly free-provider
snapshots. It is not equivalent to CRSP, Compustat, or a licensed exchange reference
database and must not be described as fully survivorship-bias free.

Alpha Vantage free-tier request limits require resumable synchronization. Each monthly
response is cached, and repeated runs continue from cached progress. Massive validation
is similarly cached.

## Identifier And Classification Limits

Permanent IDs are inferred from exchange and issuer name when an authoritative
security identifier is unavailable. Corporate name changes, reorganizations, share
classes, and mergers may require manual reconciliation.

Sector exposure uses a broad SEC SIC-to-sector mapping. It is suitable for risk
grouping but is not official GICS classification. Runs fail the sector coverage gate
when too many securities remain unclassified.

## Prices And Corporate Actions

Yahoo Finance and Stooq are not authoritative exchange feeds. Delisting returns,
corporate actions, late corrections, and historical volume may be incomplete. The
backtest exposes terminal-price sensitivity rather than hiding this uncertainty.

## Fundamentals And Macro

SEC company facts can contain restatements, duplicate contexts, taxonomy changes, and
issuer-specific tags. Filing date is used as the conservative availability date.

FRED observations are currently explanatory and do not change portfolio weights.
Historical revision vintages are not reconstructed; macro results must not be
presented as a full ALFRED vintage backtest.

## Upgrade Path

The provider interfaces allow replacement with licensed point-in-time security master,
price, delisting, sector, and fundamental datasets without changing experiment,
backtest, or report contracts.

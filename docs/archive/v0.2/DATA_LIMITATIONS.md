# Data Limitations

## Universe Source

The default system reconstructs S&P 500 point-in-time membership from the free
Wikipedia constituents table and constituent-change log. This avoids the previous
current-membership backfill workflow, but it is not a licensed S&P Dow Jones index
membership feed.

Future-dated constituent changes are reversed when they fall after the requested
backtest end date. Historical changes before the available change-log history remain
limited by the public source.

## Identifier And Classification Limits

Permanent IDs prefer SEC CIKs when available and otherwise fall back to deterministic
symbol-derived IDs. Corporate name changes, reorganizations, share classes, and
mergers may require manual reconciliation.

Sector exposure uses current S&P GICS metadata when available and SEC SIC-derived
metadata for refreshed company records. SIC-to-sector mapping is approximate and is
not official GICS classification.

## Prices And Corporate Actions

Yahoo Finance and Stooq are not authoritative exchange feeds. Some removed S&P 500
constituents have incomplete free price histories. The strict gate allows explicit
coverage gaps only when configured coverage thresholds still pass.

Delisting returns, corporate actions, late corrections, and historical volume may be
incomplete. The backtest exposes terminal-price sensitivity rather than hiding this
uncertainty.

## Fundamentals And Macro

SEC company facts can contain restatements, duplicate contexts, taxonomy changes, and
issuer-specific tags. Filing date is used as the conservative availability date, and
available date is clamped so it never precedes the reported period end.

FRED observations are currently explanatory and do not change portfolio weights.
Historical revision vintages are not reconstructed; macro results must not be
presented as a full ALFRED vintage backtest.

## Upgrade Path

The provider interfaces allow replacement with licensed point-in-time index
membership, price, delisting, sector, and fundamental datasets without changing
experiment, backtest, or report contracts.

# Reviewing munin

## Design

- A field or file kept only for the dashboard: can the dashboard derive it
  from what a run already writes?

## Semantics

References: RFC 8259 for JSON.

Failure here is a run's record lost or unreadable: a crash mid-write, two
writers on one run, a file another tool cannot parse.

- A reader sees whole records while a run writes.
- System metrics on every supported system, and a metric a system lacks.

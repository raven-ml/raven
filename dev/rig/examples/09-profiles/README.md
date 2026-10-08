# `09-profiles`

A profile holds what every device did while a function ran. This example takes
a profile of an allocation and a copy, prints its events without their times,
and writes it in Chrome's trace format.

```bash
cd dev/rig/examples/09-profiles
dune exec ./main.exe
```

It writes `profile.json` in the directory it runs in; open it in
[Perfetto](https://ui.perfetto.dev) or `chrome://tracing` to see the spans on a
timeline.

## What You'll Learn

- Taking a profile: `Profile.take f` is `f ()` and its events
- Naming the host's own work: `Profile.span`
- The events devices record: spans, copies, allocations; loads, counters and
  traces from GPU work
- Profiles cost nothing when none is taken: `Profile.enabled`
- `Profile.output_chrome_trace`

## Key Functions

| Function                           | Purpose                                 |
| ---------------------------------- | --------------------------------------- |
| `Profile.take f`                   | `f ()` and the events it recorded       |
| `Profile.span name f`              | Record `f ()` as a span of the host     |
| `Profile.output_chrome_trace oc e` | Write events for Perfetto               |

## Next Steps

Continue to [10-elf](../10-elf/).

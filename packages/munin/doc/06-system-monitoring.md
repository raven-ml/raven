# System Monitoring

The `munin.sys` library provides background system monitoring that logs CPU and
memory metrics alongside your training metrics.

## Setup

Add `munin.sys` to your dune `libraries`:

<!-- $MDX skip -->
```lisp
(executable
 (name main)
 (libraries munin munin.sys))
```

## Usage

Start a monitor after creating a session, and stop it before finishing:

<!-- $MDX skip -->
```ocaml
let () =
  let session =
    Munin.Session.start ~experiment:"train" ~name:"resnet-50" ()
  in
  let monitor = Munin_sys.start session () in

  (* ... training loop ... *)

  Munin_sys.stop monitor;
  Munin.Session.finish session ()
```

The monitor spawns a background thread that samples system and process
statistics at a fixed interval and logs them as scalar metrics.

### Configuring the interval

The default sampling interval is 15 seconds. Override it with `~interval`:

<!-- $MDX skip -->
```ocaml
let monitor = Munin_sys.start session ~interval:5.0 ()
```

The first sample is taken after one interval elapses.

## Logged metrics

All metrics use the `sys/` prefix and are defined with `~summary:`Last`, so the
final sampled value appears in run summaries.

### System-wide

| Metric key          | Description                       | Range    |
|---------------------|-----------------------------------|----------|
| `sys/cpu_user`      | User CPU percentage               | 0--100   |
| `sys/cpu_system`    | System (kernel) CPU percentage    | 0--100   |
| `sys/mem_used_pct`  | Memory usage percentage           | 0--100   |
| `sys/mem_used_gb`   | Memory used in GB                 | 0+       |

### Per-process

| Metric key          | Description                       | Range    |
|---------------------|-----------------------------------|----------|
| `sys/proc_cpu_pct`  | Process CPU percentage            | 0+       |
| `sys/proc_mem_mb`   | Process resident set size in MB   | 0+       |

## Platform support

`munin.sys` works on Linux and macOS. Platform-specific behavior:

- **Linux**: CPU counters are read from `/proc/stat`; memory from
  `/proc/meminfo`; process stats from `/proc/self/stat` and `Unix.times`.
- **macOS**: CPU and memory use Mach host statistics; process memory uses
  `task_info`. Only user/nice/system/idle CPU fields are populated.

## TUI system panel

The `munin watch` dashboard displays a system panel on the right side with
CPU, memory, process CPU, and RSS bars. This panel reads the `sys/` metrics
from the run's event log -- it does not perform its own sampling. If your run
does not use `munin.sys`, the system panel shows zeroes.

Toggle the panel with `[` or `]` in the dashboard.

## Sysstat module

The `Munin_sys` module re-exports the `Sysstat` interface, giving direct access
to stateless, poll-based sampling functions. These are useful for custom
monitoring outside the background thread.

### Cpu

<!-- $MDX skip -->
```ocaml
let prev = Munin_sys.Cpu.sample () in
(* ... wait ... *)
let next = Munin_sys.Cpu.sample () in
let stats = Munin_sys.Cpu.compute ~prev ~next in
Printf.printf "User: %.1f%%  System: %.1f%%\n" stats.user stats.system
```

`Cpu.sample_per_core` returns an array of per-core counters.

### Mem

<!-- $MDX skip -->
```ocaml
let mem = Munin_sys.Mem.sample () in
let used_gb = Int64.to_float mem.used /. 1_073_741_824. in
Printf.printf "Memory: %.1f GB used / %.1f GB total\n"
  used_gb (Int64.to_float mem.total /. 1_073_741_824.)
```

### Net

<!-- $MDX skip -->
```ocaml
let prev = Munin_sys.Net.sample () in
(* ... wait ... *)
let next = Munin_sys.Net.sample () in
let stats = Munin_sys.Net.compute ~prev ~next ~dt:1.0 in
Printf.printf "Rx: %.0f B/s  Tx: %.0f B/s\n"
  stats.rx_bytes_per_sec stats.tx_bytes_per_sec
```

### Disk_io

<!-- $MDX skip -->
```ocaml
let prev = Munin_sys.Disk_io.sample () in
(* ... wait ... *)
let next = Munin_sys.Disk_io.sample () in
let stats = Munin_sys.Disk_io.compute ~prev ~next ~dt:1.0 in
Printf.printf "Read: %.0f B/s  Write: %.0f B/s  Util: %.1f%%\n"
  stats.read_bytes_per_sec stats.write_bytes_per_sec
  stats.utilization_percent
```

### Fs

<!-- $MDX skip -->
```ocaml
let fs = Munin_sys.Fs.sample () in
let used_pct =
  Int64.to_float fs.used_bytes /. Int64.to_float fs.total_bytes *. 100.
in
Printf.printf "Disk: %.1f%% used\n" used_pct;
List.iter (fun (p : Munin_sys.Fs.partition) ->
  Printf.printf "  %s: %Ld / %Ld bytes\n"
    p.mount_point p.used_bytes p.total_bytes
) fs.partitions
```

### Proc

Current process stats:

<!-- $MDX skip -->
```ocaml
let prev = Munin_sys.Proc.Self.sample () in
(* ... wait ... *)
let next = Munin_sys.Proc.Self.sample () in
let stats = Munin_sys.Proc.Self.compute ~prev ~next ~dt:1.0 ~num_cores:None in
Printf.printf "CPU: %.1f%%  RSS: %Ld bytes\n" stats.cpu_percent stats.rss_bytes
```

Process table (all visible processes):

<!-- $MDX skip -->
```ocaml
let prev = Munin_sys.Proc.Table.sample () in
(* ... wait ... *)
let next = Munin_sys.Proc.Table.sample () in
let stats = Munin_sys.Proc.Table.compute ~prev ~next ~dt:1.0 in
List.iter (fun (p : Munin_sys.Proc.Table.stats) ->
  Printf.printf "%d  %-15s  CPU: %.1f%%  Mem: %.1f%%\n"
    p.pid p.name p.cpu_percent p.mem_percent
) (List.sort (fun a b -> compare b.cpu_percent a.cpu_percent) stats)
```

### System info

<!-- $MDX skip -->
```ocaml
let (l1, l5, l15) = Munin_sys.loadavg () in
Printf.printf "Load: %.2f %.2f %.2f\n" l1 l5 l15;
Printf.printf "Uptime: %Ld seconds\n" (Munin_sys.uptime ())
```

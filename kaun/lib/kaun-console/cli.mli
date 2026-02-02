(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Command-line interface for kaun-console.

    This executable provides a standalone tool for monitoring Kaun training
    runs. It can automatically discover the latest run or monitor a specific run
    by ID.

    {2 Usage}

    {[
      (* Monitor latest run *)
      kaun-console

      (* Monitor latest run from specific experiment *)
      kaun-console --experiment mnist

      (* Monitor specific run *)
      kaun-console --runs 2026-01-22_14-36-45_mnist

      (* Filter by tags *)
      kaun-console --tag production --tag verified
    ]}

    {2 Options}

    - [--base-dir DIR] Directory containing training runs (default: [./runs])
    - [--runs ID] Specific run ID to monitor
    - [--experiment NAME] Filter runs by experiment name
    - [--tag TAG] Filter runs by tag (can be specified multiple times)

    {2 Examples}

    {[
      (* Monitor latest mnist experiment *)
      kaun-console --experiment mnist

      (* Monitor specific run *)
      kaun-console --runs 2026-01-22_14-36-45_mnist

      (* Monitor from custom runs directory *)
      kaun-console --base-dir /path/to/runs
    ]} *)

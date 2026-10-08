(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'v term = 'v Rig_packet.term =
  | Value of 'v
  | Add of 'v term * int64
  | Shift of 'v term * int
  | Or of 'v term * int64

type 'v word = 'v Rig_packet.word =
  | Dword of int
  | W32 of 'v term
  | W64 of 'v term

type 'v t = 'v word list
type scope = Agent | System

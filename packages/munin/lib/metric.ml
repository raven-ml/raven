(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type goal = [ `Minimize | `Maximize ]
type sample = { step : int; time : float; value : float }
type series = Series.t

let length (s : series) = s.length

let check (s : series) i =
  if i < 0 || i >= s.length then
    invalid_arg
      (Printf.sprintf "Metric: index %d out of [0;%d]" i (s.length - 1))

let step (s : series) i =
  check s i;
  Array.unsafe_get s.steps i

let time (s : series) i =
  check s i;
  Float.Array.unsafe_get s.times i

let value (s : series) i =
  check s i;
  Float.Array.unsafe_get s.values i

let sample (s : series) i =
  check s i;
  {
    step = Array.unsafe_get s.steps i;
    time = Float.Array.unsafe_get s.times i;
    value = Float.Array.unsafe_get s.values i;
  }

type finite = { least : sample; greatest : sample; mean : float }

type stats = {
  count : int;
  last : sample;
  finite : finite option;
  non_finite : (int * sample) option;
}

let best goal st =
  match st.finite with
  | None -> None
  | Some f ->
      Some (match goal with `Minimize -> f.least | `Maximize -> f.greatest)

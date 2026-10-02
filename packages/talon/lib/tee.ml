(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The items pulled form a list that grows at its end. Each reader holds the
   cell of its next item, so the cells that no reader holds are collected. *)
type 'a cell = { mutable state : 'a state }

and 'a state =
  | Unread
  | Item of 'a * 'a cell
  | End
  | Raised of exn * Printexc.raw_backtrace

let v n next close =
  let first = { state = Unread } and open_ = ref n in
  let reader () =
    let at = ref first and closed = ref false in
    let rec read () =
      match !at.state with
      | Item (x, rest) ->
          at := rest;
          Some x
      | End -> None
      | Raised (exn, bt) -> Printexc.raise_with_backtrace exn bt
      | Unread ->
          let cell = !at in
          cell.state <-
            (match next () with
            | Some x -> Item (x, { state = Unread })
            | None -> End
            | exception exn -> Raised (exn, Printexc.get_raw_backtrace ()));
          read ()
    in
    let close_one () =
      if not !closed then begin
        closed := true;
        at := { state = End };
        decr open_;
        if !open_ = 0 then close ()
      end
    in
    (read, close_one)
  in
  List.init n (fun _ -> reader ())

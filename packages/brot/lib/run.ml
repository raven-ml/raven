(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type deferred = {
  normalizer : Normalizer.t;
  source : string;
  mutable alignment : Normalizer.alignment option;
}

type alignment = Known of Normalizer.alignment | Deferred of deferred
type place = Shifted of int | Fixed of { start : int; stop : int }

type frame = {
  text : string;
  literal : bool;
  place : place;
  rewrite : alignment option;
  align : alignment;
  base : int;
}

type t = {
  frames : frame array;
  frame_stop : int array;
  span_start : int array;
  span_stop : int array;
  marks : int array;
  token_table : string array;
  len_table : int array;
}

(* Working out where every byte of a normalized text came from costs about as
   much again as normalizing it, and offsets are the only thing that needs it,
   so a stretch carries the normalization until they are asked for. Every frame
   of the stretch holds the same [deferred], so the stretch is normalized once
   however many frames its added tokens and pieces cut it into. *)
let force = function
  | Known alignment -> alignment
  | Deferred deferred -> (
      match deferred.alignment with
      | Some alignment -> alignment
      | None ->
          let alignment =
            snd (Normalizer.apply_aligned deferred.normalizer deferred.source)
          in
          deferred.alignment <- Some alignment;
          alignment)

let frame_align f = force f.align
let frame_rewrite f = Option.map force f.rewrite

(* A byte range of the text a frame's spans index, in the coordinates of the
   text the tokenizer was given. *)
let span_in_input ~place ~rewrite ~align ~base ~start ~stop =
  let start, stop =
    match place with
    | Fixed { start; stop } -> (start, stop)
    | Shifted shift -> (
        match rewrite with
        | None -> (shift + start, shift + stop)
        | Some rewrite ->
            let start, stop = Normalizer.original_span rewrite ~start ~stop in
            (shift + start, shift + stop))
  in
  let start, stop = Normalizer.original_span align ~start ~stop in
  (base + start, base + stop)

(* A token of a literal span is the text of the span itself: an added token
   matched with [lstrip] or [rstrip] stands for the white space it took in too,
   and its identifier alone does not say how much. *)
let tokens run ~ids =
  let result = Array.make (Array.length ids) "" in
  let table = run.token_table in
  let span = ref 0 and at = ref 0 in
  for f = 0 to Array.length run.frames - 1 do
    let frame = Array.unsafe_get run.frames f in
    let last = Array.unsafe_get run.frame_stop f in
    while !span < last do
      let mark = Array.unsafe_get run.marks !span in
      while !at < mark do
        if frame.literal then
          let start = Array.unsafe_get run.span_start !span in
          let stop = Array.unsafe_get run.span_stop !span in
          result.(!at) <- String.sub frame.text start (stop - start)
        else result.(!at) <- table.(Array.unsafe_get ids !at);
        incr at
      done;
      incr span
    done
  done;
  result

(* The tokens of a span tile it in the order they were emitted, each covering
   the source bytes its identifier accounts for. An unknown token stands for
   whatever no identifier describes and reads as zero, so the last such token of
   a span takes the bytes up to where the ones after it begin — which is what
   makes a span holding one of them exact, the common case. Several share what
   is left, a character each and the rest to the last. *)
let offsets run ~ids =
  let result = Array.make (Array.length ids) (0, 0) in
  let lens = run.len_table in
  let span = ref 0 and at = ref 0 in
  for f = 0 to Array.length run.frames - 1 do
    let frame = Array.unsafe_get run.frames f in
    let last = Array.unsafe_get run.frame_stop f in
    let place = frame.place in
    let rewrite = frame_rewrite frame in
    let align = frame_align frame in
    let base = frame.base in
    while !span < last do
      let mark = Array.unsafe_get run.marks !span in
      let stop = Array.unsafe_get run.span_stop !span in
      let opaque = ref (-1) and tail = ref 0 in
      for i = !at to mark - 1 do
        let len = Array.unsafe_get lens (Array.unsafe_get ids i) in
        if len = 0 then begin
          opaque := i;
          tail := 0
        end
        else tail := !tail + len
      done;
      let cursor = ref (Array.unsafe_get run.span_start !span) in
      while !at < mark do
        let len = Array.unsafe_get lens (Array.unsafe_get ids !at) in
        let finish =
          if !at = mark - 1 then stop
          else if len > 0 then min stop (!cursor + len)
          else if !at = !opaque then max !cursor (stop - !tail)
          else
            min stop
              (!cursor
              + Char_class.at_len (Char_class.at frame.text !cursor ~stop))
        in
        result.(!at) <-
          span_in_input ~place ~rewrite ~align ~base ~start:!cursor ~stop:finish;
        cursor := finish;
        incr at
      done;
      incr span
    done
  done;
  result

let words run ~ids =
  let result = Array.make (Array.length ids) None in
  let at = ref 0 in
  for span = 0 to Array.length run.marks - 1 do
    let mark = Array.unsafe_get run.marks span in
    let word = Some span in
    while !at < mark do
      result.(!at) <- word;
      incr at
    done
  done;
  result

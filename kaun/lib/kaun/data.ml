(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = {
  next : unit -> 'a option;
  reset : unit -> unit;
  length : int option;
}

(* Constructors *)

let of_array a =
  let n = Array.length a in
  let i = ref 0 in
  {
    next =
      (fun () ->
        if !i >= n then None
        else
          let v = a.(!i) in
          incr i;
          Some v);
    reset = (fun () -> i := 0);
    length = Some n;
  }

let of_tensor t =
  let n = (Rune.shape t).(0) in
  let i = ref 0 in
  {
    next =
      (fun () ->
        if !i >= n then None
        else
          let v = Rune.slice [ I !i ] t in
          incr i;
          Some v);
    reset = (fun () -> i := 0);
    length = Some n;
  }

let of_tensors (x, y) =
  let nx = (Rune.shape x).(0) in
  let ny = (Rune.shape y).(0) in
  if nx <> ny then
    invalid_arg
      (Printf.sprintf "Data.of_tensors: first dimensions differ (%d vs %d)" nx
         ny);
  let n = nx in
  let i = ref 0 in
  {
    next =
      (fun () ->
        if !i >= n then None
        else
          let vx = Rune.slice [ I !i ] x in
          let vy = Rune.slice [ I !i ] y in
          incr i;
          Some (vx, vy));
    reset = (fun () -> i := 0);
    length = Some n;
  }

let of_fn n f =
  if n < 0 then
    invalid_arg (Printf.sprintf "Data.of_fn: expected n >= 0, got %d" n);
  let i = ref 0 in
  {
    next =
      (fun () ->
        if !i >= n then None
        else
          let v = f !i in
          incr i;
          Some v);
    reset = (fun () -> i := 0);
    length = Some n;
  }

let repeat n v = of_fn n (fun _ -> v)

(* Transformers *)

let map f t =
  {
    next = (fun () -> Option.map f (t.next ()));
    reset = t.reset;
    length = t.length;
  }

let batch ?(drop_last = false) n t =
  if n <= 0 then
    invalid_arg (Printf.sprintf "Data.batch: expected n > 0, got %d" n);
  let batch_len =
    Option.map (fun l -> if drop_last then l / n else (l + n - 1) / n) t.length
  in
  {
    next =
      (fun () ->
        match t.next () with
        | None -> None
        | Some first ->
            let buf = Array.make n first in
            let k = ref 1 in
            let continue = ref true in
            while !k < n && !continue do
              match t.next () with
              | Some v ->
                  buf.(!k) <- v;
                  incr k
              | None -> continue := false
            done;
            if !k < n && drop_last then None
            else if !k < n then Some (Array.sub buf 0 !k)
            else Some buf);
    reset = t.reset;
    length = batch_len;
  }

let map_batch ?drop_last n f t = map f (batch ?drop_last n t)

let shuffle key t =
  match t.length with
  | None -> invalid_arg "Data.shuffle: requires a pipeline with known length"
  | Some n ->
      let perm_tensor = Rune.Rng.permutation ~key n in
      let perm = Array.map Int32.to_int (Rune.to_array perm_tensor) in
      (* Eagerly materialize the upstream into an array *)
      let elements =
        Array.init n (fun _ ->
            match t.next () with Some v -> v | None -> assert false)
      in
      let i = ref 0 in
      {
        next =
          (fun () ->
            if !i >= n then None
            else
              let v = elements.(perm.(!i)) in
              incr i;
              Some v);
        reset = (fun () -> i := 0);
        length = Some n;
      }

(* Consumers *)

let iter f t =
  let rec loop () =
    match t.next () with
    | None -> ()
    | Some v ->
        f v;
        loop ()
  in
  loop ()

let iteri f t =
  let i = ref 0 in
  let rec loop () =
    match t.next () with
    | None -> ()
    | Some v ->
        f !i v;
        incr i;
        loop ()
  in
  loop ()

let fold f init t =
  let rec loop acc =
    match t.next () with None -> acc | Some v -> loop (f acc v)
  in
  loop init

let to_array t =
  let items = ref [] in
  iter (fun v -> items := v :: !items) t;
  Array.of_list (List.rev !items)

let rec to_seq t () =
  match t.next () with None -> Seq.Nil | Some v -> Seq.Cons (v, to_seq t)

(* Properties *)

let reset t = t.reset ()
let length t = t.length

(* Utilities *)

let stack_batch tensors = Rune.stack (Array.to_list tensors)

let prepare ?shuffle:shuffle_key ~batch_size ?(drop_last = true) (x, y) =
  let nx = (Rune.shape x).(0) in
  let ny = (Rune.shape y).(0) in
  if nx <> ny then
    invalid_arg
      (Printf.sprintf "Data.prepare: first dimensions differ (%d vs %d)" nx ny);
  if batch_size <= 0 then
    invalid_arg
      (Printf.sprintf "Data.prepare: expected batch_size > 0, got %d" batch_size);
  let indices = of_fn nx Fun.id in
  let indices =
    match shuffle_key with Some key -> shuffle key indices | None -> indices
  in
  map_batch ~drop_last batch_size
    (fun idx_arr ->
      let n = Array.length idx_arr in
      let xs = Array.init n (fun j -> Rune.slice [ I idx_arr.(j) ] x) in
      let ys = Array.init n (fun j -> Rune.slice [ I idx_arr.(j) ] y) in
      (stack_batch xs, stack_batch ys))
    indices

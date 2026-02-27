(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Error messages *)

let err_discrete_n = "Space.Discrete.create: n must be strictly positive"
let err_discrete_not = "Space.Discrete: not a discrete space"
let err_box_empty = "Space.Box.create: low cannot be empty"
let err_box_shape = "Space.Box.create: low and high must have identical lengths"
let err_box_not = "Space.Box: not a box space"
let err_mb_n = "Space.Multi_binary.create: n must be strictly positive"
let err_md_empty = "Space.Multi_discrete.create: nvec must not be empty"
let err_seq_min = "Space.Sequence.create: min_length must be non-negative"
let err_seq_max = "Space.Sequence.create: max_length must be >= min_length"
let err_text_max = "Space.Text.create: max_length must be positive"
let err_text_charset = "Space.Text.create: charset must not be empty"
let strf = Printf.sprintf
let errorf fmt = Format.kasprintf (fun msg -> Error msg) fmt

(* Spec *)

type spec =
  | Discrete of { start : int; n : int }
  | Box of { low : float array; high : float array }
  | Multi_binary of { n : int }
  | Multi_discrete of { nvec : int array }
  | Tuple of spec list
  | Dict of (string * spec) list
  | Sequence of { min_length : int; max_length : int option; base : spec }
  | Text of { charset : string; max_length : int }

let rec equal_spec a b =
  match (a, b) with
  | Discrete a, Discrete b -> a.start = b.start && a.n = b.n
  | Box a, Box b -> a.low = b.low && a.high = b.high
  | Multi_binary a, Multi_binary b -> a.n = b.n
  | Multi_discrete a, Multi_discrete b -> a.nvec = b.nvec
  | Tuple a, Tuple b ->
      List.length a = List.length b && List.for_all2 equal_spec a b
  | Dict a, Dict b ->
      List.length a = List.length b
      && List.for_all2
           (fun (ka, sa) (kb, sb) -> String.equal ka kb && equal_spec sa sb)
           a b
  | Sequence a, Sequence b ->
      a.min_length = b.min_length
      && a.max_length = b.max_length
      && equal_spec a.base b.base
  | Text a, Text b ->
      String.equal a.charset b.charset && a.max_length = b.max_length
  | ( ( Discrete _ | Box _ | Multi_binary _ | Multi_discrete _ | Tuple _
      | Dict _ | Sequence _ | Text _ ),
      _ ) ->
      false

(* Space type *)

type 'a t = {
  spec : spec;
  shape : int array option;
  contains : 'a -> bool;
  sample : unit -> 'a;
  pack : 'a -> Value.t;
  unpack : Value.t -> ('a, string) result;
  boundaries : Value.t list;
  box_bounds : (float array * float array) option;
  discrete_info : (int * int) option;
}

type packed = Pack : 'a t -> packed

let spec s = s.spec
let shape s = s.shape
let contains s v = s.contains v
let sample s = s.sample ()
let pack s v = s.pack v
let unpack s v = s.unpack v
let boundary_values s = s.boundaries

(* Discrete *)

module Discrete = struct
  type element = (int32, Nx.int32_elt) Nx.t

  let to_int tensor =
    let reshaped = Nx.reshape [| 1 |] tensor in
    let arr : Int32.t array = Nx.to_array reshaped in
    Int32.to_int arr.(0)

  let of_int v = Nx.scalar Nx.int32 (Int32.of_int v)

  let create ?(start = 0) n =
    if n <= 0 then invalid_arg err_discrete_n;
    let hi = start + n in
    let contains tensor =
      let reshaped = Nx.reshape [| 1 |] tensor in
      let arr : Int32.t array = Nx.to_array reshaped in
      Array.length arr = 1
      &&
      let v = Int32.to_int arr.(0) in
      v >= start && v < hi
    in
    let sample () =
      let tensor = Nx.randint Nx.int32 ~high:hi [| 1 |] start in
      let arr : Int32.t array = Nx.to_array tensor in
      Nx.scalar Nx.int32 arr.(0)
    in
    let pack tensor =
      let arr : Int32.t array = Nx.to_array (Nx.reshape [| 1 |] tensor) in
      Value.Int (Int32.to_int arr.(0))
    in
    let unpack = function
      | Value.Int v when v >= start && v < hi ->
          Ok (Nx.scalar Nx.int32 (Int32.of_int v))
      | Value.Int v -> errorf "Discrete value %d outside [%d, %d)" v start hi
      | other -> errorf "Discrete expects Int, got %s" (Value.to_string other)
    in
    let boundaries =
      if n = 1 then [ Value.Int start ]
      else [ Value.Int start; Value.Int (hi - 1) ]
    in
    {
      spec = Discrete { start; n };
      shape = None;
      contains;
      sample;
      pack;
      unpack;
      boundaries;
      box_bounds = None;
      discrete_info = Some (start, n);
    }

  let n s =
    match s.discrete_info with
    | Some (_, n) -> n
    | None -> invalid_arg err_discrete_not

  let start s =
    match s.discrete_info with
    | Some (start, _) -> start
    | None -> invalid_arg err_discrete_not
end

(* Box *)

module Box = struct
  type element = (float, Nx.float32_elt) Nx.t

  let create ~low ~high =
    let arity = Array.length low in
    if arity = 0 then invalid_arg err_box_empty;
    if arity <> Array.length high then invalid_arg err_box_shape;
    Array.iteri
      (fun i lo ->
        if lo > high.(i) then
          invalid_arg
            (strf "Space.Box.create: low[%d]=%g > high[%d]=%g" i lo i high.(i)))
      low;
    let low = Array.copy low in
    let high = Array.copy high in
    let contains tensor =
      let sh = Nx.shape tensor in
      Array.length sh = 1
      && sh.(0) = arity
      &&
      let values = Nx.to_array tensor in
      let rec loop i =
        if i = arity then true
        else
          let v = values.(i) in
          v >= low.(i) && v <= high.(i) && loop (i + 1)
      in
      loop 0
    in
    let sample () =
      let uniform = Nx.rand Nx.float32 [| arity |] in
      let draws = Nx.to_array uniform in
      let values =
        Array.init arity (fun i ->
            let lo = low.(i) in
            let hi = high.(i) in
            if Float.equal lo hi then lo
            else
              let range = hi -. lo in
              if Float.is_finite range then lo +. (draws.(i) *. range)
              else
                let v = -1e6 +. (draws.(i) *. 2e6) in
                Float.max lo (Float.min hi v))
      in
      Nx.create Nx.float32 [| arity |] values
    in
    let pack tensor = Value.Float_array (Array.copy (Nx.to_array tensor)) in
    let unpack = function
      | Value.Float_array arr when Array.length arr = arity ->
          let tensor = Nx.create Nx.float32 [| arity |] arr in
          if contains tensor then Ok tensor
          else
            errorf "Box value outside bounds: %s"
              (Value.to_string (Value.Float_array arr))
      | Value.Float_array arr ->
          errorf "Box expects vector of size %d, got size %d" arity
            (Array.length arr)
      | other ->
          errorf "Box expects Float_array, got %s" (Value.to_string other)
    in
    let identical =
      let same = ref true in
      let i = ref 0 in
      while !same && !i < arity do
        if not (Float.equal low.(!i) high.(!i)) then same := false;
        incr i
      done;
      !same
    in
    let boundaries =
      let lo_v = Value.Float_array (Array.copy low) in
      let hi_v = Value.Float_array (Array.copy high) in
      if identical then [ lo_v ] else [ lo_v; hi_v ]
    in
    let box_bounds = Some (Array.copy low, Array.copy high) in
    {
      spec = Box { low = Array.copy low; high = Array.copy high };
      shape = Some [| arity |];
      contains;
      sample;
      pack;
      unpack;
      boundaries;
      box_bounds;
      discrete_info = None;
    }

  let bounds s =
    match s.box_bounds with
    | Some (low, high) -> (Array.copy low, Array.copy high)
    | None -> invalid_arg err_box_not
end

(* Multi_binary *)

module Multi_binary = struct
  type element = (int32, Nx.int32_elt) Nx.t

  let create n =
    if n <= 0 then invalid_arg err_mb_n;
    let contains tensor =
      let sh = Nx.shape tensor in
      Array.length sh = 1
      && sh.(0) = n
      &&
      let arr : Int32.t array = Nx.to_array tensor in
      Array.for_all (fun v -> v = Int32.zero || v = Int32.one) arr
    in
    let sample () = Nx.randint Nx.int32 ~high:2 [| n |] 0 in
    let pack tensor =
      let arr : Int32.t array = Nx.to_array tensor in
      Value.Bool_array
        (Array.init n (fun i -> not (Int32.equal arr.(i) Int32.zero)))
    in
    let unpack = function
      | Value.Bool_array arr when Array.length arr = n ->
          let data =
            Array.map (fun b -> if b then Int32.one else Int32.zero) arr
          in
          Ok (Nx.create Nx.int32 [| n |] data)
      | Value.Bool_array arr ->
          errorf "Multi_binary expects vector of size %d, got size %d" n
            (Array.length arr)
      | other ->
          errorf "Multi_binary expects Bool_array, got %s"
            (Value.to_string other)
    in
    let boundaries =
      [
        Value.Bool_array (Array.make n false);
        Value.Bool_array (Array.make n true);
      ]
    in
    {
      spec = Multi_binary { n };
      shape = Some [| n |];
      contains;
      sample;
      pack;
      unpack;
      boundaries;
      box_bounds = None;
      discrete_info = None;
    }
end

(* Multi_discrete *)

module Multi_discrete = struct
  type element = (int32, Nx.int32_elt) Nx.t

  let create nvec =
    let arity = Array.length nvec in
    if arity = 0 then invalid_arg err_md_empty;
    let nvec = Array.copy nvec in
    Array.iteri
      (fun i bound ->
        if bound <= 0 then
          invalid_arg
            (strf "Space.Multi_discrete.create: nvec[%d] must be > 0" i))
      nvec;
    let contains tensor =
      let sh = Nx.shape tensor in
      Array.length sh = 1
      && sh.(0) = arity
      &&
      let arr : Int32.t array = Nx.to_array tensor in
      let rec loop i =
        if i = arity then true
        else
          let v = Int32.to_int arr.(i) in
          v >= 0 && v < nvec.(i) && loop (i + 1)
      in
      loop 0
    in
    let sample () =
      let data =
        Array.init arity (fun i ->
            let tensor = Nx.randint Nx.int32 ~high:nvec.(i) [| 1 |] 0 in
            let arr = Nx.to_array tensor in
            arr.(0))
      in
      Nx.create Nx.int32 [| arity |] data
    in
    let pack tensor =
      let arr : Int32.t array = Nx.to_array tensor in
      Value.Int_array (Array.map Int32.to_int arr)
    in
    let unpack = function
      | Value.Int_array arr when Array.length arr = arity ->
          let data = Array.map Int32.of_int arr in
          let tensor = Nx.create Nx.int32 [| arity |] data in
          if contains tensor then Ok tensor
          else
            errorf "Multi_discrete value outside bounds: %s"
              (Value.to_string (Value.Int_array arr))
      | Value.Int_array arr ->
          errorf "Multi_discrete expects vector of size %d, got size %d" arity
            (Array.length arr)
      | other ->
          errorf "Multi_discrete expects Int_array, got %s"
            (Value.to_string other)
    in
    let boundaries =
      [
        Value.Int_array (Array.make arity 0);
        Value.Int_array (Array.init arity (fun i -> nvec.(i) - 1));
      ]
    in
    {
      spec = Multi_discrete { nvec = Array.copy nvec };
      shape = Some [| arity |];
      contains;
      sample;
      pack;
      unpack;
      boundaries;
      box_bounds = None;
      discrete_info = None;
    }
end

(* Tuple *)

module Tuple = struct
  type element = Value.t list

  let create spaces =
    let spaces = Array.of_list spaces in
    let len = Array.length spaces in
    let contains values =
      let rec loop i = function
        | [] -> i = len
        | v :: rest -> (
            if i >= len then false
            else
              let (Pack s) = spaces.(i) in
              match s.unpack v with
              | Ok _ -> loop (i + 1) rest
              | Error _ -> false)
      in
      loop 0 values
    in
    let sample () =
      let values =
        Array.to_list
          (Array.init len (fun i ->
               let (Pack s) = spaces.(i) in
               let v = s.sample () in
               s.pack v))
      in
      values
    in
    let pack values = Value.List values in
    let unpack = function
      | Value.List values ->
          if List.length values <> len then
            errorf "Tuple expects %d elements, got %d" len (List.length values)
          else
            let rec loop i = function
              | [] -> Ok values
              | v :: rest -> (
                  let (Pack s) = spaces.(i) in
                  match s.unpack v with
                  | Ok _ -> loop (i + 1) rest
                  | Error msg -> errorf "Tuple element %d: %s" i msg)
            in
            loop 0 values
      | other -> errorf "Tuple expects List, got %s" (Value.to_string other)
    in
    let sub_specs = Array.to_list (Array.map (fun (Pack s) -> s.spec) spaces) in
    {
      spec = Tuple sub_specs;
      shape = None;
      contains;
      sample;
      pack;
      unpack;
      boundaries = [];
      box_bounds = None;
      discrete_info = None;
    }
end

(* Dict *)

module Dict = struct
  type element = (string * Value.t) list

  module String_map = Map.Make (String)

  let create entries =
    let map =
      List.fold_left
        (fun acc (key, space) ->
          if String_map.mem key acc then
            invalid_arg (strf "Space.Dict.create: duplicate key '%s'" key);
          String_map.add key space acc)
        String_map.empty entries
    in
    let contains values =
      let rec loop remaining m =
        match remaining with
        | [] -> String_map.is_empty m
        | (key, value) :: rest -> (
            match String_map.find_opt key m with
            | None -> false
            | Some (Pack s) -> (
                match s.unpack value with
                | Ok _ -> loop rest (String_map.remove key m)
                | Error _ -> false))
      in
      loop values map
    in
    let sample () =
      if String_map.is_empty map then []
      else
        let acc =
          String_map.fold
            (fun key (Pack s) acc ->
              let v = s.sample () in
              (key, s.pack v) :: acc)
            map []
        in
        List.rev acc
    in
    let pack values = Value.Dict values in
    let unpack = function
      | Value.Dict values ->
          if contains values then Ok values
          else errorf "Dict contains unexpected keys or values"
      | other -> errorf "Dict expects Dict, got %s" (Value.to_string other)
    in
    let sub_specs =
      List.rev
        (String_map.fold (fun key (Pack s) acc -> (key, s.spec) :: acc) map [])
    in
    {
      spec = Dict sub_specs;
      shape = None;
      contains;
      sample;
      pack;
      unpack;
      boundaries = [];
      box_bounds = None;
      discrete_info = None;
    }
end

(* Sequence *)

module Sequence = struct
  type 'a element = 'a list

  let create ?(min_length = 0) ?max_length base =
    if min_length < 0 then invalid_arg err_seq_min;
    let max_length =
      match max_length with
      | None -> None
      | Some m when m < min_length -> invalid_arg err_seq_max
      | Some _ as m -> m
    in
    let contains values =
      let len = List.length values in
      len >= min_length
      && (match max_length with None -> true | Some m -> len <= m)
      && List.for_all (fun v -> base.contains v) values
    in
    let sample () =
      let length =
        match max_length with
        | None -> min_length
        | Some max_len ->
            if max_len = min_length then min_length
            else
              let tensor =
                Nx.randint Nx.int32 ~high:(max_len + 1) [| 1 |] min_length
              in
              let arr = Nx.to_array tensor in
              Int32.to_int arr.(0)
      in
      if length = 0 then []
      else
        let rec build i acc =
          if i = length then List.rev acc
          else
            let v = base.sample () in
            build (i + 1) (v :: acc)
        in
        build 0 []
    in
    let pack values = Value.List (List.map (fun v -> base.pack v) values) in
    let unpack = function
      | Value.List values ->
          let len = List.length values in
          let exceeds =
            match max_length with None -> false | Some m -> len > m
          in
          if len < min_length || exceeds then
            match max_length with
            | None ->
                errorf "Sequence length %d shorter than minimum %d" len
                  min_length
            | Some m ->
                errorf "Sequence length %d outside [%d, %d]" len min_length m
          else
            let rec loop acc = function
              | [] -> Ok (List.rev acc)
              | v :: rest -> (
                  match base.unpack v with
                  | Ok x -> loop (x :: acc) rest
                  | Error _ as err -> err)
            in
            loop [] values
      | other -> errorf "Sequence expects List, got %s" (Value.to_string other)
    in
    {
      spec = Sequence { min_length; max_length; base = base.spec };
      shape = None;
      contains;
      sample;
      pack;
      unpack;
      boundaries = [];
      box_bounds = None;
      discrete_info = None;
    }
end

(* Text *)

module Text = struct
  type element = string

  let default_charset =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

  let create ?(charset = default_charset) ?(max_length = 64) () =
    if max_length <= 0 then invalid_arg err_text_max;
    let charset_len = String.length charset in
    if charset_len = 0 then invalid_arg err_text_charset;
    let contains value =
      let len = String.length value in
      len <= max_length
      &&
      let rec loop i =
        if i = len then true
        else String.contains charset value.[i] && loop (i + 1)
      in
      loop 0
    in
    let sample () =
      let length =
        if max_length = 1 then 1
        else
          let tensor = Nx.randint Nx.int32 ~high:(max_length + 1) [| 1 |] 1 in
          let arr = Nx.to_array tensor in
          Int32.to_int arr.(0)
      in
      if length = 0 then ""
      else
        let idxs = Nx.randint Nx.int32 ~high:charset_len [| length |] 0 in
        let arr = Nx.to_array idxs in
        Bytes.init length (fun i -> charset.[Int32.to_int arr.(i)])
        |> Bytes.to_string
    in
    let pack value = Value.String value in
    let unpack = function
      | Value.String s when contains s -> Ok s
      | Value.String s -> errorf "Text value '%s' violates constraints" s
      | other -> errorf "Text expects String, got %s" (Value.to_string other)
    in
    let example = if charset_len = 0 then "" else String.make 1 charset.[0] in
    let boundaries = [ Value.String ""; Value.String example ] in
    {
      spec = Text { charset; max_length };
      shape = None;
      contains;
      sample;
      pack;
      unpack;
      boundaries;
      box_bounds = None;
      discrete_info = None;
    }
end

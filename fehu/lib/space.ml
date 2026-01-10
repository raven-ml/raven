(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Format

module Value = struct
  type t =
    | Int of int
    | Float of float
    | Bool of bool
    | Int_array of int array
    | Float_array of float array
    | Bool_array of bool array
    | List of t list
    | Tuple of t list
    | Dict of (string * t) list
    | String of string

  let rec pp fmt = function
    | Int i -> fprintf fmt "%d" i
    | Float f -> fprintf fmt "%g" f
    | Bool b -> fprintf fmt "%B" b
    | Int_array arr -> pp_array fmt (fun fmt x -> fprintf fmt "%d" x) arr
    | Float_array arr -> pp_array fmt (fun fmt x -> fprintf fmt "%g" x) arr
    | Bool_array arr -> pp_array fmt (fun fmt x -> fprintf fmt "%B" x) arr
    | List values -> fprintf fmt "[%a]" pp_list values
    | Tuple values -> fprintf fmt "(%a)" pp_list values
    | Dict entries -> fprintf fmt "{%a}" pp_dict entries
    | String s -> fprintf fmt "%S" s

  and pp_array : type a.
      formatter -> (formatter -> a -> unit) -> a array -> unit =
   fun fmt pp_elem arr ->
    fprintf fmt "[|";
    for idx = 0 to Array.length arr - 1 do
      if idx > 0 then fprintf fmt "; ";
      pp_elem fmt arr.(idx)
    done;
    fprintf fmt "|]"

  and pp_list fmt = function
    | [] -> ()
    | x :: xs ->
        pp fmt x;
        List.iter (fun value -> fprintf fmt ", %a" pp value) xs

  and pp_dict fmt entries =
    let rec loop first = function
      | [] -> ()
      | (key, value) :: rest ->
          if not first then fprintf fmt "; ";
          fprintf fmt "%s=%a" key pp value;
          loop false rest
    in
    loop true entries

  let to_string value = asprintf "%a" pp value
end

open Value

module Space_table = Hashtbl.Make (struct
  type t = Obj.t

  let equal a b = a == b
  let hash = Hashtbl.hash
end)

let boundary_registry : Value.t list Space_table.t = Space_table.create 64

let box_bounds_registry : (float array * float array) Space_table.t =
  Space_table.create 32

let register_boundary_values space values =
  match values with
  | [] -> ()
  | _ -> Space_table.replace boundary_registry (Obj.repr space) values

let boundary_values space =
  match Space_table.find_opt boundary_registry (Obj.repr space) with
  | Some values -> values
  | None -> []

let register_box_bounds space low high =
  Space_table.replace box_bounds_registry (Obj.repr space) (low, high)

let box_bounds_for space =
  Space_table.find_opt box_bounds_registry (Obj.repr space)

let default_rng rng = Option.value rng ~default:(Rune.Rng.key 42)

let take_key rng =
  let keys = Rune.Rng.split rng in
  (keys.(0), keys.(1))

type 'a t = {
  shape : int array option;
  contains : 'a -> bool;
  sample : ?rng:Rune.Rng.key -> unit -> 'a * Rune.Rng.key;
  pack : 'a -> Value.t;
  unpack : Value.t -> ('a, string) result;
}

type packed = Pack : 'a t -> packed

let shape space = space.shape
let contains space value = space.contains value
let sample ?rng space = space.sample ?rng ()
let pack space value = space.pack value
let unpack space value = space.unpack value
let errorf fmt = Format.kasprintf (fun msg -> Error msg) fmt

module Discrete = struct
  type element = (int32, Rune.int32_elt) Rune.t

  let create ?(start = 0) n =
    if n <= 0 then
      invalid_arg "Space.Discrete.create: n must be strictly positive";
    let contains tensor =
      let reshaped = Rune.reshape [| 1 |] tensor in
      let values : Int32.t array = Rune.to_array reshaped in
      Array.length values = 1
      &&
      let v = Int32.to_int values.(0) in
      v >= start && v < start + n
    in
    let space =
      {
        shape = None;
        contains;
        sample =
          (fun ?rng () ->
            let rng = default_rng rng in
            let sample_key, next_rng = take_key rng in
            let tensor =
              Rune.randint Rune.int32 ~key:sample_key ~high:(start + n) [| 1 |]
                start
            in
            let values : Int32.t array = Rune.to_array tensor in
            (Rune.scalar Rune.int32 values.(0), next_rng));
        pack =
          (fun tensor ->
            let values : Int32.t array =
              Rune.to_array (Rune.reshape [| 1 |] tensor)
            in
            Int (Int32.to_int values.(0)));
        unpack =
          (function
          | Int value when value >= start && value < start + n ->
              Ok (Rune.scalar Rune.int32 (Int32.of_int value))
          | Int value ->
              errorf "Discrete value %d outside of [%d, %d)" value start
                (start + n)
          | other ->
              errorf "Discrete expects Int value, received %s"
                (Value.to_string other));
      }
    in
    let boundaries =
      if n = 1 then [ Int start ] else [ Int start; Int (start + n - 1) ]
    in
    register_boundary_values space boundaries;
    space
end

module Box = struct
  type element = (float, Rune.float32_elt) Rune.t

  let guard_bounds low high =
    Array.iteri
      (fun idx low_i ->
        let high_i = high.(idx) in
        if low_i > high_i then
          invalid_arg
            (Printf.sprintf
               "Space.Box.create: low[%d]=%f greater than high[%d]=%f" idx low_i
               idx high_i))
      low

  let create ~low ~high =
    if Array.length low = 0 then
      invalid_arg "Space.Box.create: low cannot be empty";
    if Array.length low <> Array.length high then
      invalid_arg "Space.Box.create: low and high must have identical shapes";
    guard_bounds low high;
    let low = Array.copy low in
    let high = Array.copy high in
    let arity = Array.length low in
    let contains tensor =
      let shape = Rune.shape tensor in
      Array.length shape = 1
      && shape.(0) = arity
      &&
      let values = Rune.to_array tensor in
      let rec loop idx =
        if idx = arity then true
        else
          let v = values.(idx) in
          let l = low.(idx) in
          let h = high.(idx) in
          if v < l || v > h then false else loop (idx + 1)
      in
      loop 0
    in
    let space =
      {
        shape = Some [| arity |];
        contains;
        sample =
          (fun ?rng () ->
            let rng = default_rng rng in
            let sample_key, next_rng = take_key rng in
            let uniform = Rune.rand Rune.float32 ~key:sample_key [| arity |] in
            let draws = Rune.to_array uniform in
            let values =
              Array.init arity (fun idx ->
                  let low_i = low.(idx) in
                  let high_i = high.(idx) in
                  if Float.equal low_i high_i then low_i
                  else
                    let range = high_i -. low_i in
                    low_i +. (draws.(idx) *. range))
            in
            (Rune.create Rune.float32 [| arity |] values, next_rng));
        pack =
          (fun tensor ->
            let values = Rune.to_array tensor in
            Float_array (Array.copy values));
        unpack =
          (function
          | Float_array arr when Array.length arr = arity ->
              let tensor = Rune.create Rune.float32 [| arity |] arr in
              if contains tensor then Ok tensor
              else
                errorf "Box value outside of bounds: %s"
                  (Value.to_string (Float_array arr))
          | Float_array arr ->
              errorf "Box expects vector of size %d, received size %d" arity
                (Array.length arr)
          | other ->
              errorf "Box expects Float_array, received %s"
                (Value.to_string other));
      }
    in
    let low_copy = Array.copy low in
    let high_copy = Array.copy high in
    register_box_bounds space low_copy high_copy;
    let identical_bounds =
      let same = ref true in
      let idx = ref 0 in
      while !same && !idx < arity do
        if not (Float.equal low_copy.(!idx) high_copy.(!idx)) then same := false;
        incr idx
      done;
      !same
    in
    let boundary_values =
      let low_value = Float_array (Array.copy low_copy) in
      let high_value = Float_array (Array.copy high_copy) in
      if identical_bounds then [ low_value ] else [ low_value; high_value ]
    in
    register_boundary_values space boundary_values;
    space

  let bounds space =
    match box_bounds_for space with
    | Some (low, high) -> (Array.copy low, Array.copy high)
    | None ->
        invalid_arg
          "Space.Box.bounds: unknown box space; ensure it was created via \
           Space.Box.create"
end

module Multi_binary = struct
  type element = (int32, Rune.int32_elt) Rune.t

  let create n =
    if n <= 0 then invalid_arg "Space.Multi_binary.create: n must be positive";
    let space =
      {
        shape = Some [| n |];
        contains =
          (fun tensor ->
            let shape = Rune.shape tensor in
            Array.length shape = 1
            && shape.(0) = n
            &&
            let arr : Int32.t array = Rune.to_array tensor in
            Array.for_all (fun v -> v = Int32.zero || v = Int32.one) arr);
        sample =
          (fun ?rng () ->
            let rng = default_rng rng in
            let sample_key, next_rng = take_key rng in
            let tensor =
              Rune.randint Rune.int32 ~key:sample_key ~high:2 [| n |] 0
            in
            (tensor, next_rng));
        pack =
          (fun tensor ->
            let arr : Int32.t array = Rune.to_array tensor in
            Bool_array
              (Array.init n (fun idx -> not (Int32.equal arr.(idx) Int32.zero))));
        unpack =
          (function
          | Bool_array arr when Array.length arr = n ->
              let data =
                Array.map (fun b -> if b then Int32.one else Int32.zero) arr
              in
              Ok (Rune.create Rune.int32 [| n |] data)
          | Bool_array arr ->
              errorf "MultiBinary expects vector of size %d, received size %d" n
                (Array.length arr)
          | other ->
              errorf "MultiBinary expects Bool_array, received %s"
                (Value.to_string other));
      }
    in
    let zero = Array.make n false in
    let one = Array.make n true in
    register_boundary_values space
      [ Bool_array (Array.copy zero); Bool_array (Array.copy one) ];
    space
end

module Multi_discrete = struct
  type element = (int32, Rune.int32_elt) Rune.t

  let create bounds =
    if Array.length bounds = 0 then
      invalid_arg "Space.Multi_discrete.create: empty bounds";
    Array.iteri
      (fun idx bound ->
        if bound <= 0 then
          invalid_arg
            (Format.sprintf
               "Space.Multi_discrete.create: bounds[%d] must be > 0" idx))
      bounds;
    let arity = Array.length bounds in
    let contains tensor =
      let shape = Rune.shape tensor in
      Array.length shape = 1
      && shape.(0) = arity
      &&
      let arr : Int32.t array = Rune.to_array tensor in
      let rec loop idx =
        if idx = arity then true
        else
          let max_value = bounds.(idx) in
          let v = Int32.to_int arr.(idx) in
          if v < 0 || v >= max_value then false else loop (idx + 1)
      in
      loop 0
    in
    let space =
      {
        shape = Some [| arity |];
        contains;
        sample =
          (fun ?rng () ->
            let rng = default_rng rng in
            let sample_key, next_rng = take_key rng in
            let keys =
              if arity = 0 then [||] else Rune.Rng.split ~n:arity sample_key
            in
            let data =
              Array.init arity (fun idx ->
                  let tensor =
                    Rune.randint Rune.int32 ~key:keys.(idx) ~high:bounds.(idx)
                      [| 1 |] 0
                  in
                  let arr = Rune.to_array tensor in
                  arr.(0))
            in
            (Rune.create Rune.int32 [| arity |] data, next_rng));
        pack =
          (fun tensor ->
            let arr : Int32.t array = Rune.to_array tensor in
            Int_array (Array.map Int32.to_int arr));
        unpack =
          (function
          | Int_array arr when Array.length arr = arity ->
              let data = Array.map Int32.of_int arr in
              let tensor = Rune.create Rune.int32 [| arity |] data in
              if contains tensor then Ok tensor
              else
                errorf "MultiDiscrete value outside of bounds: %s"
                  (Value.to_string (Int_array arr))
          | Int_array arr ->
              errorf "MultiDiscrete expects vector of size %d, received size %d"
                arity (Array.length arr)
          | other ->
              errorf "MultiDiscrete expects Int_array, received %s"
                (Value.to_string other));
      }
    in
    let zeros = Array.make arity 0 in
    let maxes = Array.init arity (fun idx -> max 0 (bounds.(idx) - 1)) in
    register_boundary_values space
      [ Int_array (Array.copy zeros); Int_array (Array.copy maxes) ];
    space
end

module Tuple = struct
  type element = Value.t list

  let create spaces =
    let spaces = Array.of_list spaces in
    let expected_length = Array.length spaces in
    let contains values =
      let rec loop idx = function
        | [] -> idx = expected_length
        | value :: tail -> (
            if idx >= expected_length then false
            else
              let (Pack space) = spaces.(idx) in
              match space.unpack value with
              | Ok _ -> loop (idx + 1) tail
              | Error _ -> false)
      in
      loop 0 values
    in
    {
      shape = None;
      contains;
      sample =
        (fun ?rng () ->
          let rng = default_rng rng in
          let sample_key, next_rng = take_key rng in
          let keys =
            if expected_length = 0 then [||]
            else Rune.Rng.split ~n:expected_length sample_key
          in
          let values =
            Array.init expected_length (fun idx ->
                let (Pack space) = spaces.(idx) in
                let sample_value, _ = space.sample ~rng:keys.(idx) () in
                space.pack sample_value)
            |> Array.to_list
          in
          (values, next_rng));
      pack = (fun values -> Tuple values);
      unpack =
        (function
        | Tuple values ->
            if List.length values <> expected_length then
              errorf "Tuple expects %d elements, received %d" expected_length
                (List.length values)
            else Ok values
        | other ->
            errorf "Tuple expects tuple value, received %s"
              (Value.to_string other));
    }
end

module Dict = struct
  type element = (string * Value.t) list

  module String_map = Map.Make (String)

  let create entries =
    let map =
      entries |> List.to_seq
      |> Seq.fold_left
           (fun acc (key, Pack space) ->
             if String_map.mem key acc then
               invalid_arg
                 (Printf.sprintf "Space.Dict.create: duplicate key '%s'" key);
             String_map.add key (Pack space) acc)
           String_map.empty
    in
    let contains values =
      let rec loop remaining map =
        match remaining with
        | [] -> String_map.is_empty map
        | (key, value) :: rest -> (
            match String_map.find_opt key map with
            | None -> false
            | Some (Pack space) -> (
                match space.unpack value with
                | Ok _ -> loop rest (String_map.remove key map)
                | Error _ -> false))
      in
      loop values map
    in
    {
      shape = None;
      contains;
      sample =
        (fun ?rng () ->
          let rng = default_rng rng in
          let sample_key, next_rng = take_key rng in
          let cardinal = String_map.cardinal map in
          if cardinal = 0 then ([], next_rng)
          else
            let keys = Rune.Rng.split ~n:cardinal sample_key in
            let index = ref 0 in
            let acc =
              String_map.fold
                (fun key (Pack space) acc ->
                  let sample_value, _ = space.sample ~rng:keys.(!index) () in
                  incr index;
                  (key, space.pack sample_value) :: acc)
                map []
            in
            (List.rev acc, next_rng));
      pack = (fun values -> Dict values);
      unpack =
        (function
        | Dict values ->
            if contains values then Ok values
            else errorf "Dict contains unexpected keys or values"
        | other ->
            errorf "Dict expects object value, received %s"
              (Value.to_string other));
    }
end

module Sequence = struct
  type 'a element = 'a list

  let create ?(min_length = 0) ?max_length base =
    if min_length < 0 then
      invalid_arg "Space.Sequence.create: min_length must be non-negative";
    let max_length =
      match max_length with
      | None -> None
      | Some max_length when max_length < min_length ->
          invalid_arg "Space.Sequence.create: max_length must be >= min_length"
      | Some max_length -> Some max_length
    in
    let contains values =
      let len = List.length values in
      let within_upper =
        match max_length with None -> true | Some max_len -> len <= max_len
      in
      len >= min_length && within_upper
      && List.for_all (fun value -> base.contains value) values
    in
    {
      shape = None;
      contains;
      sample =
        (fun ?rng () ->
          let rng = default_rng rng in
          let sample_key, next_rng = take_key rng in
          let length_key, elements_seed = take_key sample_key in
          let length =
            match max_length with
            | None -> min_length
            | Some max_len ->
                if max_len = min_length then min_length
                else
                  let tensor =
                    Rune.randint Rune.int32 ~key:length_key ~high:(max_len + 1)
                      [| 1 |] min_length
                  in
                  let arr = Rune.to_array tensor in
                  Int32.to_int arr.(0)
          in
          let values =
            if length = 0 then []
            else
              let keys = Rune.Rng.split ~n:length elements_seed in
              let rec build idx acc =
                if idx = length then List.rev acc
                else
                  let sample_value, _ = base.sample ~rng:keys.(idx) () in
                  build (idx + 1) (sample_value :: acc)
              in
              build 0 []
          in
          (values, next_rng));
      pack =
        (fun values ->
          Value.List (List.map (fun value -> base.pack value) values));
      unpack =
        (function
        | Value.List values ->
            let len = List.length values in
            let exceeds =
              match max_length with
              | None -> false
              | Some max_len -> len > max_len
            in
            if len < min_length || exceeds then
              match max_length with
              | None ->
                  errorf "Sequence length %d shorter than minimum %d" len
                    min_length
              | Some max_len ->
                  errorf "Sequence length %d outside of [%d, %d]" len min_length
                    max_len
            else
              let rec loop acc = function
                | [] -> Ok (List.rev acc)
                | value :: rest -> (
                    match base.unpack value with
                    | Ok v -> loop (v :: acc) rest
                    | Error _ as err -> err)
              in
              loop [] values
        | other ->
            errorf "Sequence expects list value, received %s"
              (Value.to_string other));
    }
end

module Text = struct
  type element = string

  let default_charset =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

  let create ?charset ?(max_length = 64) () =
    if max_length <= 0 then
      invalid_arg "Space.Text.create: max_length must be positive";
    let charset = Option.value charset ~default:default_charset in
    let charset_length = String.length charset in
    if charset_length = 0 then
      invalid_arg "Space.Text.create: charset must not be empty";
    let contains value =
      let len = String.length value in
      len <= max_length
      &&
      let rec loop idx =
        if idx = len then true
        else
          let ch = value.[idx] in
          if String.contains charset ch then loop (idx + 1) else false
      in
      loop 0
    in
    let space =
      {
        shape = None;
        contains;
        sample =
          (fun ?rng () ->
            let rng = default_rng rng in
            let sample_key, next_rng = take_key rng in
            let length_key, chars_key = take_key sample_key in
            let length =
              if max_length = 1 then 1
              else
                let tensor =
                  Rune.randint Rune.int32 ~key:length_key ~high:(max_length + 1)
                    [| 1 |] 1
                in
                let arr = Rune.to_array tensor in
                Int32.to_int arr.(0)
            in
            let str =
              if length = 0 then ""
              else
                let idxs =
                  Rune.randint Rune.int32 ~key:chars_key ~high:charset_length
                    [| length |] 0
                in
                let arr = Rune.to_array idxs in
                Bytes.init length (fun idx -> charset.[Int32.to_int arr.(idx)])
                |> Bytes.to_string
            in
            (str, next_rng));
        pack = (fun value -> Value.String value);
        unpack =
          (function
          | Value.String s when contains s -> Ok s
          | Value.String s -> errorf "Text value '%s' violates constraints" s
          | other ->
              errorf "Text expects string value, received %s"
                (Value.to_string other));
      }
    in
    let example =
      if charset_length = 0 then "" else String.make 1 charset.[0]
    in
    register_boundary_values space [ Value.String ""; Value.String example ];
    space
end

(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

(* Storage *)

let rec with_storage x dt =
  match (op x, arg x) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p ->
      replace x ~arg:(Param { p with dtype = dt })
  | _ ->
      let storage = with_storage (nth x 0) dt in
      replace x ~src:(storage :: List.tl (src x))

(* Estimates *)

module Estimates = struct
  type t = estimates

  let zero = { ops = Int 0; lds = Int 0; mem = Int 0 }

  let ssimplify_sint = function Int n -> Int n | Sym u -> ssimplify u

  let simplify e =
    let s = ssimplify_sint in
    { ops = s e.ops; lds = s e.lds; mem = s e.mem }

  let is o u = Op.equal (op u) o

  (* hardware indices are already counted in the multiplier *)
  let without_specials = function
    | Int n -> Int n
    | Sym m ->
        let zero x = (x, const_like x (`Int Bigint.zero)) in
        Sym
          (substitute ~calls:Skip ~pass:Fixed_point m
             (List.map zero
                (List.filter (is Op.Special) (toposort ~calls:Skip m))))

  (* Counts are exact: a range bounded by a value loaded from memory runs up to
     its type's greatest count, whose products outgrow an [int]. An integer
     result is an [Int] where it fits one, else a constant node. *)
  module X = struct
    let of_z z =
      if Bigint.fits_int z then Int (Bigint.to_int z) else Sym (const (`Int z))

    let exact fz fs (a : sint) (b : sint) =
      match (a, b) with
      | Int x, Int y -> of_z (fz (Bigint.of_int x) (Bigint.of_int y))
      | _ -> fs a b

    let ( * ) = exact Bigint.mul Sint.( * )
    let ( + ) = exact Bigint.add Sint.( + )

    let of_value = function
      | `Int z -> of_z z
      | v -> Sym (const (v :> Dtype.const))
  end

  let add e o = X.{ ops = e.ops + o.ops; lds = e.lds + o.lds; mem = e.mem + o.mem }

  let of_uops ?(ignore_indexing = false) uops =
    let flops = ref (Int 0) and lds = ref (Int 0) and mults = ref (Int 1) in
    (* The ALU operations counted since [mults] last changed, which join
       [flops] as one product: under a symbolic multiplier, such as a split
       loop's, [ssimplify] then folds one term per loop body. *)
    let work = ref Bigint.zero in
    let flush () =
      if not (Bigint.equal !work Bigint.zero) then begin
        flops := X.(!flops + (!mults * of_z !work));
        work := Bigint.zero
      end
    in
    let mem = ref [] and mult_stack = ref [] and excluded = Tbl.create 16 in
    if ignore_indexing then
      List.iter
        (fun u ->
          if is Op.Index u || is Op.Shrink u then
            let gate x = not (is Op.End x || is Op.Backedge x) in
            toposort ~calls:Skip ~gate (sink (List.tl (src u)))
            |> List.iter (fun x -> Tbl.replace excluded x ()))
        uops;
    let counted u = not (Tbl.mem excluded u) in
    let bytes u dt = Int (max_numel u * Dtype.itemsize dt) in
    let times n = X.(n * !mults) in
    let access buf o =
      let same ((b, o'), _) = b == buf && Op.equal o o' in
      match List.find_opt same !mem with
      | Some (_, total) -> total
      | None ->
          let total = ref (Int 0) in
          mem := !mem @ [ ((buf, o), total) ];
          total
    in
    List.iter
      (fun u ->
        let o = op u in
        (if is Op.Load u || is Op.Store u then
           let rec storage b =
             match src b with
             | s :: _ when not (Op.Set.mem (op b) Op.Set.defines) -> storage s
             | _ -> b
           in
           let buf = storage u in
           if is Op.Param buf then
             (* capped at the buffer's size, for re-reads such as a matmul's *)
             let total = access buf o and idx = nth u 0 in
             let accessed = X.(!total + times (bytes idx (dtype idx))) in
             total := smin [ accessed; bytes buf (dtype buf) ]);
        match o with
        | Op.Range ->
            flush ();
            mult_stack := !mults :: !mult_stack;
            if not (Dtype.equal (dtype u) Dtype.Void) then
              (* a bound that reads other ranges or memory counts its greatest
                 value: a count reads the run's variables alone *)
              let bound = Shape.simplify (without_after (nth u 0)) in
              let n =
                if known_at_launch u then ssimplify bound
                else X.of_value (vmax bound)
              in
              mults := without_specials X.(!mults * n)
        | Op.End | Op.Backedge -> (
            flush ();
            match !mult_stack with
            | m :: rest ->
                mults := m;
                mult_stack := rest
            | [] -> invalid_arg "an END or BACKEDGE closes no RANGE")
        | Op.Special ->
            flush ();
            mults := X.(!mults * ssimplify (nth u 0))
        | Op.Load when addrspace (nth u 0) <> Some Dtype.Reg ->
            lds := X.(!lds + times (bytes u (dtype u)))
        | Op.Store when addrspace (nth u 0) <> Some Dtype.Reg ->
            lds := X.(!lds + times (bytes u (dtype (nth u 1))))
        | Op.Wmma when counted u -> (
            match arg u with
            | Wmma { dims = n, m, k; threads; _ } ->
                let per_thread = 2 * n * m * k / threads in
                flush ();
                flops := X.(!flops + times (Int per_thread))
            | _ -> invalid_arg "a WMMA without its tensor core argument")
        | o when Op.Set.mem o Op.Set.alu && counted u ->
            let per = if Op.equal o Op.Mulacc then 2 else 1 in
            work := Bigint.add !work (Bigint.of_int (per * max_numel u))
        | _ -> ())
      uops;
    let mem = List.fold_left (fun s (_, t) -> X.(s + !t)) (Int 0) !mem in
    flush ();
    { ops = ssimplify_sint !flops; lds = !lds; mem }
end

(* Compilers *)

module Compiler = struct
  exception Compile_error of string

  type t = {
    cachekey : (unit -> string) option;
    compile : string list -> string list;
    disassemble : string -> unit;
  }

  (* [once f] is [f ()], computed at the first call that returns, by one domain
     at a time. *)
  let once f =
    let lock = Mutex.create () and value = ref None in
    fun () ->
      Mutex.protect lock @@ fun () ->
      match !value with
      | Some v -> v
      | None ->
          let v = f () in
          value := Some v;
          v

  let batch ?cachekey ?(disassemble = ignore) compile =
    { cachekey = Option.map once cachekey; compile; disassemble }

  let v ?cachekey ?disassemble compile =
    batch ?cachekey ?disassemble (List.map compile)

  let cachekey c = Option.map (fun key -> key ()) c.cachekey

  let compile c src =
    match c.compile [ src ] with
    | [ lib ] -> lib
    | _ -> invalid_arg "a compiler returns one binary per source"

  let disassemble c lib = c.disassemble lib

  (* Binaries are kept in the table the compiler's key names while the setting
     ccache holds. An entry that does not read, as a damaged one, or that
     cannot be read, is compiled anew and replaced. *)
  let table c = if Setting.value Setting.ccache then cachekey c else None

  let kept c src =
    let read table =
      try Helpers.Diskcache.get ~table src with Failure _ | Sys_error _ -> None
    in
    Option.bind (table c) read

  (* The sources not kept are compiled in one call, each once. When that call
     raises, each is compiled alone, in order, so that the first that raises is
     the one reported, as [List.map] would, and a call that failed for none of
     them still compiles them. *)
  let compile_all c srcs =
    let table = table c in
    let found = List.map (fun src -> (src, kept c src)) srcs in
    let missing =
      List.sort_uniq String.compare
        (List.filter_map
           (function src, None -> Some src | _, Some _ -> None)
           found)
    in
    if missing <> [] && Setting.value Setting.assert_compile then
      invalid_arg
        ("tried to compile with ASSERT_COMPILE set\n"
        ^ fst (List.find (fun (_, lib) -> Option.is_none lib) found));
    let libs =
      if missing = [] then []
      else if List.compare_length_with missing 1 = 0 then c.compile missing
      else
        try c.compile missing
        with Compile_error _ ->
          let alone = Hashtbl.create 8 in
          List.iter
            (fun (src, lib) ->
              if Option.is_none lib && not (Hashtbl.mem alone src) then
                Hashtbl.replace alone src (compile c src))
            found;
          List.map (Hashtbl.find alone) missing
    in
    if List.compare_lengths libs missing <> 0 then
      invalid_arg "a compiler returns one binary per source";
    let made = Hashtbl.create (List.length missing) in
    List.iter2
      (fun src lib ->
        Hashtbl.replace made src lib;
        Option.iter (fun table -> Helpers.Diskcache.put ~table src lib) table)
      missing libs;
    List.map
      (fun (src, lib) ->
        match lib with Some lib -> lib | None -> Hashtbl.find made src)
      found

  let compile_cached c src =
    match compile_all c [ src ] with
    | [ lib ] -> lib
    | _ -> invalid_arg "a compiler returns one binary per source"
end

(* Renderer *)

type registers = { count : int; bytes : int }

type t = {
  name : string;
  target : Helpers.Target.t;
  suffix : string;
  supports_float4 : bool;
  half8 : bool;
  registers : unit -> registers option;
  vector_lanes : int;
  has_local : bool;
  has_shared : bool;
  global_max : int list;
  local_max : int list;
  global_prod_max : int list option;
  shared_max : int;
  async_copy : bool;
  load_matrix : bool;
  tensor_cores : Tc.t list;
  per_byte_first : int;
  stream_threads : int;
  local_workgroups : int;
  extra_matcher : (unit, Ops.t) Pattern_matcher.t;
  packed_args : bool;
  code_for_op : (Op.t * (string list -> Dtype.t -> string)) list;
  native : Dtype.t -> bool;
  render : Ops.t list -> string;
  compiler : Compiler.t;
}

(* Ops.SPECIAL indexes are int32 *)
let int32_max = [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ]

let v ?(name = "Renderer") ?(suffix = "") ?(supports_float4 = true)
    ?(half8 = false)
    ?(registers = fun () -> None) ?(vector_lanes = 1) ?(has_local = true)
    ?(has_shared = true) ?(global_max = int32_max) ?(local_max = int32_max)
    ?global_prod_max ?(shared_max = 32768) ?(async_copy = false)
    ?(load_matrix = false)
    ?(tensor_cores = [])
    ?(per_byte_first = 64) ?(stream_threads = 32768)
    ?(local_workgroups = 256)
    ?(extra_matcher = Pattern_matcher.v (fun () -> [])) ?(packed_args = false)
    ?(code_for_op = [])
    ?(native = fun _ -> true)
    ?(render = fun _ -> invalid_arg "needs a renderer")
    ?(compiler = Compiler.v Fun.id) target =
  {
    name;
    target;
    suffix;
    supports_float4;
    half8;
    registers;
    vector_lanes;
    has_local;
    has_shared;
    global_max;
    local_max;
    global_prod_max;
    shared_max;
    async_copy;
    load_matrix;
    tensor_cores;
    per_byte_first;
    stream_threads;
    local_workgroups;
    extra_matcher;
    packed_args;
    code_for_op;
    native;
    render;
    compiler;
  }

let with_compiler compiler r = { r with compiler }

let emulated () =
  let dtype n =
    match Dtype.of_string n with Ok dt -> dt | Error e -> invalid_arg e
  in
  List.map dtype (Setting.value Setting.emulated_dtypes)

(* double can't be bitcast to anything without long support *)
let supported_dtypes r =
  let no_double = Dtype.mem Dtype.Int64 (emulated ()) in
  List.filter
    (fun dt -> r.native dt && not (no_double && Dtype.equal dt Dtype.Float64))
    Dtype.all

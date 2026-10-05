(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

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

  let add e o =
    Sint.{ ops = e.ops + o.ops; lds = e.lds + o.lds; mem = e.mem + o.mem }

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
          (substitute m
             (List.map zero (List.filter (is Op.Special) (toposort m))))

  let of_uops ?(ignore_indexing = false) uops =
    let flops = ref (Int 0) and lds = ref (Int 0) and mults = ref (Int 1) in
    let mem = ref [] and mult_stack = ref [] and excluded = Tbl.create 16 in
    if ignore_indexing then
      List.iter
        (fun u ->
          if is Op.Index u || is Op.Shrink u then
            let gate x = not (is Op.End x || is Op.Backedge x) in
            toposort ~gate (sink (List.tl (src u)))
            |> List.iter (fun x -> Tbl.replace excluded x ()))
        uops;
    let counted u = not (Tbl.mem excluded u) in
    let bytes u dt = Int (max_numel u * Dtype.itemsize dt) in
    let times n = Sint.(n * !mults) in
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
             | s :: _ when not (is Op.Param b) -> storage s
             | _ -> b
           in
           let buf = storage u in
           if is Op.Param buf then
             (* capped at the buffer's size, for re-reads such as a matmul's *)
             let total = access buf o and idx = nth u 0 in
             let accessed = Sint.(!total + times (bytes idx (dtype idx))) in
             total := smin [ accessed; bytes buf (dtype buf) ]);
        match o with
        | Op.Range ->
            mult_stack := !mults :: !mult_stack;
            if not (Dtype.equal (dtype u) Dtype.Void) then
              mults := without_specials Sint.(!mults * ssimplify (nth u 0))
        | Op.End | Op.Backedge -> (
            match !mult_stack with
            | m :: rest ->
                mults := m;
                mult_stack := rest
            | [] -> invalid_arg "an END or BACKEDGE closes no RANGE")
        | Op.Special -> mults := Sint.(!mults * ssimplify (nth u 0))
        | Op.Load when addrspace (nth u 0) <> Some Dtype.Reg ->
            lds := Sint.(!lds + times (bytes u (dtype u)))
        | Op.Store when addrspace (nth u 0) <> Some Dtype.Reg ->
            lds := Sint.(!lds + times (bytes u (dtype (nth u 1))))
        | Op.Wmma when counted u -> (
            match arg u with
            | Wmma { dims = n, m, k; threads; _ } ->
                let per_thread = 2 * n * m * k / threads in
                flops := Sint.(!flops + times (Int per_thread))
            | _ -> invalid_arg "a WMMA without its tensor core argument")
        | o when Op.Set.mem o Op.Set.alu && counted u ->
            let per = Int (if Op.equal o Op.Mulacc then 2 else 1) in
            let n = Int (max_numel u) in
            flops := Sint.(!flops + (!mults * per * n))
        | _ -> ())
      uops;
    let mem = List.fold_left (fun s (_, t) -> Sint.(s + !t)) (Int 0) !mem in
    { ops = ssimplify_sint !flops; lds = !lds; mem }
end

(* Compilers *)

module Compiler = struct
  exception Compile_error of string

  type t = {
    cachekey : (unit -> string) option;
    compile : string -> string;
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

  let v ?cachekey ?(disassemble = ignore) compile =
    { cachekey = Option.map once cachekey; compile; disassemble }

  let cachekey c = Option.map (fun key -> key ()) c.cachekey
  let compile c src = c.compile src
  let disassemble c lib = c.disassemble lib

  (* Binaries are kept in the table the compiler's key names while the setting
     ccache holds. An entry that does not read, as a damaged one, is compiled
     anew and replaced. *)
  let compile_cached c src =
    let table = if Setting.value Setting.ccache then cachekey c else None in
    let kept table =
      try Helpers.Diskcache.get ~table src with Failure _ -> None
    in
    match Option.bind table kept with
    | Some lib -> lib
    | None ->
        if Setting.value Setting.assert_compile then
          invalid_arg ("tried to compile with ASSERT_COMPILE set\n" ^ src);
        let lib = c.compile src in
        Option.iter (fun table -> Helpers.Diskcache.put ~table src lib) table;
        lib
end

(* Renderer *)

type t = {
  name : string;
  target : Helpers.Target.t;
  suffix : string;
  supports_float4 : bool;
  has_local : bool;
  has_shared : bool;
  global_max : int list;
  local_max : int list;
  global_prod_max : int list option;
  shared_max : int;
  tensor_cores : Tc.t list;
  extra_matcher : (unit, Ops.t) Pattern_matcher.t;
  code_for_op : (Op.t * (string list -> Dtype.t -> string)) list;
  native : Dtype.t -> bool;
  render : Ops.t list -> string;
  compiler : Compiler.t;
}

(* Ops.SPECIAL indexes are int32 *)
let int32_max = [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ]

let v ?(name = "Renderer") ?(suffix = "") ?(supports_float4 = true)
    ?(has_local = true) ?(has_shared = true) ?(global_max = int32_max)
    ?(local_max = int32_max) ?global_prod_max ?(shared_max = 32768)
    ?(tensor_cores = []) ?(extra_matcher = Pattern_matcher.v (fun () -> []))
    ?(code_for_op = []) ?(native = fun _ -> true)
    ?(render = fun _ -> invalid_arg "needs a renderer")
    ?(compiler = Compiler.v Fun.id) target =
  {
    name;
    target;
    suffix;
    supports_float4;
    has_local;
    has_shared;
    global_max;
    local_max;
    global_prod_max;
    shared_max;
    tensor_cores;
    extra_matcher;
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
  let no_double = List.exists (Dtype.equal Dtype.Int64) (emulated ()) in
  List.filter
    (fun dt -> r.native dt && not (no_double && Dtype.equal dt Dtype.Float64))
    Dtype.all

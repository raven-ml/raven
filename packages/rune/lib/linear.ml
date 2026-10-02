(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

(* A tape's entries, one per slot. A linear call's outputs are consecutive
   slots: the first holds the call, the others [Part]. *)
type entry =
  | Input
  | Recorded : ('a, 'b) Nx.t Nx.Op.t -> entry
  | Call of {
      inputs : Nx.packed list;
      like : Nx.packed list;
      pullback : Nx.packed list -> Nx.packed list;
    }
  | Part

type tape = {
  entry : string;
  mutable entries : entry array;
  mutable length : int;
}

type (_, _) Repr.node +=
  | Slot : { tape : tape; index : int } -> ('a, 'b) Repr.node

let real_or_complex dt = Nx_dtype.is_float dt || Nx_dtype.is_complex dt
let differentiable x = real_or_complex (Nx.dtype x)

(* A complex element is the pair of its components, so a bitcast between a
   complex dtype and the float of its components reads the same real
   coordinates, as does one to a real or complex operand's own dtype. Any other
   bitcast reads a value's bits as an unrelated value. *)
let same_coordinates : type a b c d.
    (a, b) Nx_dtype.t -> (c, d) Nx_dtype.t -> bool =
 fun src dst ->
  match (src, dst) with
  | Nx_dtype.Complex64, Nx_dtype.Float32 | Float32, Complex64 -> true
  | Complex128, Float64 | Float64, Complex128 -> true
  | _ -> real_or_complex src && Nx_dtype.equal src dst

let create entry = { entry; entries = Array.make 64 Input; length = 0 }

let owns t x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { tape; _ } -> tape == t
      | _ -> false)
  | Host _ | Placed _ -> false

let index x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { index; _ } -> index
      | _ -> assert false)
  | Host _ | Placed _ -> assert false

let slot t ~context placement dtype shape e =
  if t.length = Array.length t.entries then begin
    let entries = Array.make (2 * t.length) Input in
    Array.blit t.entries 0 entries 0 t.length;
    t.entries <- entries
  end;
  let index = t.length in
  t.entries.(index) <- e;
  t.length <- index + 1;
  Repr.Traced.v ~context placement dtype shape (Slot { tape = t; index })

let input t x =
  slot t ~context:(Repr.context x) (Nx.placement x) (Nx.dtype x) (Nx.shape x)
    Input

let call t inputs pullback like =
  List.mapi
    (fun j (Nx.P y) ->
      let e = if j = 0 then Call { inputs; like; pullback } else Part in
      Nx.P
        (slot t ~context:(Repr.context y) (Nx.placement y) (Nx.dtype y)
           (Nx.shape y) e))
    like

(* Recording *)

let nonlinear t op =
  invalid_arg
    (Printf.sprintf
       "%s: a custom_jvp tangent map applies %s to a tangent; a tangent map \
        must be linear in its tangents"
       t.entry (name op))

(* [record t op x] is the slot of [op]'s result, recorded on [t]; [x] is a slot
   operand, whose context the result takes. *)
let record t op x =
  slot t ~context:(Repr.context x) (placement op) (Nx.Op.dtype op)
    (Nx.Op.shape op) (Recorded op)

(* [record_any t op] is [record t op x] for the first slot operand [x]. A slot's
   context reaches only the constants the frontend makes beside it, which a
   trace inlines, so which slot operand gives it does not show. *)
let record_any t op =
  let (Nx.P x) = List.find (fun (Nx.P x) -> owns t x) (operands op) in
  record t op x

let sum t op (k : Nx_backend.reduce) x =
  match[@warning "@4@8"] k with
  | Sum -> record t op x
  | Prod | Max | Min -> nonlinear t op

(* [run t op] is [op], one of whose operands is a slot of [t]: the slot of its
   result if [op] is linear in its slots. A plain operand of an operation linear
   in several is taken as zero, as a tangent's zero fill is. *)
let run : type r. tape -> r Nx.Op.t -> r =
 fun t op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> (
      match[@warning "@4@8"] (k : Nx_backend.unary) with
      | Neg -> record t op x
      | Recip | Abs | Sqrt | Sign | Exp | Log | Log1p | Expm1 | Sin | Cos | Tan
      | Asin | Acos | Atan | Sinh | Cosh | Tanh | Trunc | Ceil | Floor | Round
      | Erf ->
          nonlinear t op)
  | Binary (k, a, b) ->
      let sa = owns t a and sb = owns t b in
      let linear =
        match[@warning "@4@8"] (k : Nx_backend.binary) with
        | Add | Sub -> sa && sb
        | Mul -> sa <> sb
        | Fdiv -> not sb
        | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor -> false
      in
      if linear then record_any t op else nonlinear t op
  | Where _ -> record_any t op
  | Fma (a, b, c) ->
      if owns t c && owns t a <> owns t b then record t op c else nonlinear t op
  | Reduce (k, _, x) -> sum t op k x
  | Scan (k, _, x) -> sum t op k x
  | Pad (_, v, x) ->
      if v = Nx_dtype.zero (Nx.dtype x) then record t op x else nonlinear t op
  | Convert (Cast, dtype, x) ->
      if real_or_complex dtype && real_or_complex (Nx.dtype x) then
        record t op x
      else nonlinear t op
  | Convert (Bitcast, dtype, x) ->
      if same_coordinates (Nx.dtype x) dtype then record t op x
      else nonlinear t op
  | Solve_triangular { a; b; _ } ->
      if owns t a then nonlinear t op else record t op b
  | Matmul (a, b) ->
      if owns t a && owns t b then nonlinear t op else record_any t op
  | Cat _ -> record_any t op
  | Gather _ -> record_any t op
  | Scatter { mode = `Set | `Add; _ } -> record_any t op
  | Scatter { mode = `Max | `Min; _ } -> nonlinear t op
  | Update _ -> record_any t op
  | Unfold _ -> record_any t op
  | Fold _ -> record_any t op
  | Fft _ -> record_any t op
  | Rfft _ -> record_any t op
  | Irfft _ -> record_any t op
  | Contiguous _ -> record_any t op
  | Move _ -> record_any t op
  | Place _ -> record_any t op
  | Read { by; _ } ->
      invalid_arg
        (Printf.sprintf
           "%s: a custom_jvp tangent map reads a tangent's value with %s; \
            under reverse mode a tangent has none"
           t.entry by)
  | Compare _ | Arg_reduce _ | Sort _ | Argsort _ | Threefry _ | Cholesky _
  | Qr _ | Lu _ | Svd _ | Eig _ | Eigh _ | Check _ ->
      nonlinear t op

(* Gathering across the lanes of the map named [axis] is linear: its transpose
   is the calling lane's row of the sum of every lane's cotangent, a
   reduce-scatter. *)
let lanes t axis x =
  let n = Construct.perform (Lane_count axis) in
  let shape = Nx.shape x in
  let like =
    Nx.broadcast_to
      (Array.append [| n |] shape)
      (Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros_like x))
  in
  let pullback = function
    | [ Nx.P ct ] ->
        let ct = Nx.unpack (Nx.dtype x) (Nx.P ct) in
        let summed =
          Nx.sum ~axes:[ 0 ] (Construct.perform (Lanes (axis, ct)))
        in
        let index = Construct.perform (Lane_index (Some axis)) in
        [
          Nx.P
            (Nx.reshape shape
               (Nx.take ~axis:0
                  ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 index))
                  summed));
        ]
    | _ -> assert false (* One output. *)
  in
  match call t [ Nx.P x ] pullback [ Nx.P like ] with
  | [ y ] -> Nx.unpack (Nx.dtype x) y
  | _ -> assert false (* One output. *)

let rec answer : type r. tape -> r Construct.t -> (unit -> r) option =
 fun t c ->
  match[@warning "@4@8"] c with
  | Detach x -> if owns t x then Some (fun () -> x) else None
  | Add (_, v) ->
      if owns t v then
        Some
          (fun () ->
            invalid_arg
              "Rune.Total.add: a custom_jvp tangent map adds a tangent under \
               reverse mode; a total takes values")
      else None
  | Lanes (axis, x) ->
      if owns t x then Some (fun () -> lanes t axis x) else None
  | Compiled { p; f; args; _ } ->
      (* A tangent reaches a compiled call only through a custom_jvp tangent
         map, whose contract makes it linear in its tangents: the call runs
         under the tape, which records its operations. *)
      if Nx.Ptree.fold p (fun _ x any -> any || owns t x) args false then
        Some (fun () -> install t (fun () -> f args))
      else None
  | Scan _ | Remat _ | Barrier _ | Custom _ | Lane_index _ | Lane_count _ ->
      None

and install : type a. tape -> (unit -> a) -> a =
 fun t f ->
  let owner = { Construct.owns = (fun x -> owns t x) } in
  let claims op = Construct.claims owner op in
  Construct.install
    {
      op = Some { run = (fun op -> run t op); claims };
      call = (fun c -> answer t c);
    }
    f

(* Transposing *)

type cotangents = { tape : tape; cts : Nx.packed option array }

let cotangents t = { tape = t; cts = Array.make t.length None }

let add cts x ct =
  if owns cts.tape x then begin
    if Nx.shape ct <> Nx.shape x then
      invalid_arg
        (Format.asprintf
           "Rune: a cotangent of shape %a reached a value of shape %a"
           Nx.pp_shape (Nx.shape ct) Nx.pp_shape (Nx.shape x));
    let i = index x in
    cts.cts.(i) <-
      Some
        (match cts.cts.(i) with
        | None -> Nx.P ct
        | Some prev -> Nx.P (Nx.add (Nx.unpack (Nx.dtype ct) prev) ct))
  end

let cotangent cts x =
  if owns cts.tape x then Option.map (Nx.unpack (Nx.dtype x)) cts.cts.(index x)
  else None

(* [unbroadcast ct shape] is [ct] summed over the axes along which [shape] was
   broadcast to [ct]'s shape. *)
let unbroadcast ct shape =
  let ct_shape = Nx.shape ct in
  if ct_shape = shape then ct
  else
    let lead = Array.length ct_shape - Array.length shape in
    let axes =
      List.filter
        (fun i -> i < lead || (shape.(i - lead) = 1 && ct_shape.(i) <> 1))
        (List.init (Array.length ct_shape) Fun.id)
    in
    Nx.reshape shape (Nx.sum ~axes ct)

(* [shrink_axis ~axis (lo, hi) x] is [x] from [lo] to [hi] along [axis];
   [pad_axis ~axis (lo, hi) x] is [x] with [lo] and [hi] zeros around it along
   [axis]. *)
let shrink_axis ~axis (lo, hi) x =
  Nx.shrink
    (Array.mapi (fun i d -> if i = axis then (lo, hi) else (0, d)) (Nx.shape x))
    x

let pad_axis ~axis (lo, hi) x =
  Nx.pad
    (Array.mapi (fun i _ -> if i = axis then (lo, hi) else (0, 0)) (Nx.shape x))
    (Nx_dtype.zero (Nx.dtype x))
    x

(* The cotangent of the scattered updates: the cotangent at the positions they
   reach, less, under [`Set] with repeated indices, the updates a later one
   overwrites. *)
let scattered ~mode ~unique ~axis ~indices ~into ct =
  let g = Nx.take_along_axis ~axis ~indices ct in
  match mode with
  | `Set when not unique ->
      let shape = Nx.shape indices in
      let along =
        Array.mapi (fun i _ -> if i = axis then shape.(axis) else 1) shape
      in
      let rank =
        Nx.broadcast_to shape
          (Nx.reshape along (Nx.arange Nx.int64 0 shape.(axis) 1))
      in
      let winner =
        eval
          (Scatter
             {
               mode = `Set;
               unique = false;
               axis;
               indices;
               updates = rank;
               into = Nx.zeros Nx.int64 (Nx.shape into);
             })
      in
      Nx.where
        (Nx.equal (Nx.take_along_axis ~axis ~indices winner) rank)
        g (Nx.zeros_like g)
  | `Set | `Add -> g

(* The window of [ct] that [v], written at [starts], covers; each axis is read
   with a gather, so a traced [starts] stays traced. *)
let window ~starts v ct =
  let vshape = Nx.shape v in
  let rank = Array.length vshape in
  let w = ref ct in
  for axis = 0 to rank - 1 do
    let len = vshape.(axis) in
    let start = Nx.reshape [||] (Nx.slice [ Nx.I axis ] starts) in
    let idx = Nx.add (Nx.arange Nx.int64 0 len 1) start in
    let shape = Array.copy (Nx.shape !w) in
    shape.(axis) <- len;
    let along = Array.init rank (fun i -> if i = axis then len else 1) in
    w :=
      Nx.take_along_axis ~axis
        ~indices:(Nx.broadcast_to shape (Nx.reshape along idx))
        !w
  done;
  !w

let transpose_op : type a b.
    cotangents -> (a, b) Nx.t Nx.Op.t -> (a, b) Nx.t -> unit =
 fun cts op ct ->
  let owns x = owns cts.tape x and add x v = add cts x v in
  match[@warning "@4@8"] op with
  | Unary (_, x) -> add x (Nx.neg ct)
  | Binary (k, a, b) -> (
      match[@warning "@4@8"] (k : Nx_backend.binary) with
      | Add ->
          add a ct;
          add b ct
      | Sub ->
          add a ct;
          add b (Nx.neg ct)
      | Mul -> if owns a then add a (Nx.mul ct b) else add b (Nx.mul a ct)
      | Fdiv -> add a (Nx.div ct b)
      | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor ->
          assert false (* Never recorded. *))
  | Where (c, a, b) ->
      let zeros = Nx.zeros_like ct in
      add a (Nx.where c ct zeros);
      add b (Nx.where c zeros ct)
  | Fma (a, b, c) ->
      if owns a then add a (Nx.mul ct b) else add b (Nx.mul a ct);
      add c ct
  | Reduce (_, axes, x) ->
      let shape = Nx.shape x in
      let kept =
        Array.mapi (fun i d -> if Array.mem i axes then 1 else d) shape
      in
      add x (Nx.broadcast_to shape (Nx.reshape kept ct))
  | Scan (_, axis, x) ->
      add x
        (Nx.flip ~axes:[ axis ] (Nx.cumsum ~axis (Nx.flip ~axes:[ axis ] ct)))
  | Pad (padding, _, x) ->
      add x
        (Nx.shrink
           (Array.mapi (fun i (lo, _) -> (lo, lo + (Nx.shape x).(i))) padding)
           ct)
  | Cat (axis, xs) ->
      ignore
        (List.fold_left
           (fun lo x ->
             let hi = lo + (Nx.shape x).(axis) in
             add x (shrink_axis ~axis (lo, hi) ct);
             hi)
           0 xs)
  | Convert (Cast, _, x) -> add x (Nx.cast (Nx.dtype x) ct)
  | Convert (Bitcast, _, x) ->
      (* The tape holds a complex cotangent conjugated, [dL/dre - i dL/dim], and
         a pair of float components holds [dL/dre] and [dL/dim], so a bitcast
         between the two conjugates its complex side. [Nx.conjugate] leaves the
         real side as it is. *)
      let back ct = Nx.bitcast (Nx.dtype x) ct in
      if Nx_dtype.is_complex (Nx.dtype x) = Nx_dtype.is_complex (Nx.dtype ct)
      then add x (back ct)
      else add x (Nx.conjugate (back (Nx.conjugate ct)))
  | Gather (axis, indices, x) ->
      add x
        (eval
           (Scatter
              {
                mode = `Add;
                unique = false;
                axis;
                indices;
                updates = ct;
                into = Nx.zeros_like x;
              }))
  | Scatter
      { mode = (`Set | `Add) as mode; unique; axis; indices; updates; into } ->
      add updates (scattered ~mode ~unique ~axis ~indices ~into ct);
      add into
        (match mode with
        | `Add -> ct
        | `Set ->
            Nx.mul ct
              (eval
                 (Scatter
                    {
                      mode = `Set;
                      unique;
                      axis;
                      indices;
                      updates = Nx.zeros_like updates;
                      into = Nx.ones_like into;
                    })))
  | Scatter { mode = `Max | `Min; _ } -> assert false (* Never recorded. *)
  | Update (x, starts, v) ->
      add x (eval (Update (ct, starts, Nx.zeros_like v)));
      add v (window ~starts v ct)
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      let shape = Nx.shape x and k = Array.length kernel_size in
      let output_size = Array.sub shape (Array.length shape - k) k in
      add x
        (eval
           (Fold { output_size; kernel_size; stride; dilation; padding; x = ct }))
  | Fold { kernel_size; stride; dilation; padding; x; _ } ->
      add x (eval (Unfold { kernel_size; stride; dilation; padding; x = ct }))
  | Fft { inverse; axes; x } -> add x (eval (Fft { inverse; axes; x = ct }))
  | Rfft { axes; x; _ } ->
      (* rfft is a real embedding, an fft and a slice to the first n/2 + 1 bins
         of the last axis: each transposes to its adjoint, a zero pad, the fft
         itself and the real part. *)
      let last = axes.(Array.length axes - 1) in
      let n = (Nx.shape x).(last) and m = (Nx.shape ct).(last) in
      let ct =
        if (n > m) [@mutate off "a pad by zero leaves the cotangent"] then
          pad_axis ~axis:last (0, n - m) ct
        else ct
      in
      add x
        (Nx.real (Nx.dtype x) (eval (Fft { inverse = false; axes; x = ct })))
  | Irfft { axes; x; _ } ->
      (* irfft resizes the spectrum along the last axis to the n / 2 + 1 bins
         its output length n reads, truncating or padding with zeros, extends it
         by its conjugate mirror, runs the inverse fft and takes the real part.
         The transpose embeds the real cotangent, runs the same inverse fft,
         folds the mirror back (bins 1 .. n - m received a second contribution,
         which on a real cotangent's transform equals the first) and undoes the
         resize, padding where it truncated and truncating where it padded. An
         output of no element reads no bin. *)
      let last = axes.(Array.length axes - 1) in
      let n = (Nx.shape ct).(last) and bins = (Nx.shape x).(last) in
      if n > 0 then begin
        let m = (n / 2) + 1 in
        let z =
          eval (Fft { inverse = true; axes; x = Nx.cast (Nx.dtype x) ct })
        in
        let head = shrink_axis ~axis:last (0, m) z in
        let folded =
          if
            (n - m >= 1)
            [@mutate off "at n = m the mirror is empty and the pad adds zeros"]
          then
            Nx.add head
              (pad_axis ~axis:last
                 (1, m - 1 - (n - m))
                 (shrink_axis ~axis:last (1, n - m + 1) head))
          else head
        in
        add x
          (if
             (m > bins)
             [@mutate off "a shrink to every bin leaves the spectrum"]
           then shrink_axis ~axis:last (0, bins) folded
           else if (m < bins) [@mutate off "a pad by zero leaves the spectrum"]
           then pad_axis ~axis:last (0, bins - m) folded
           else folded)
      end
  | Contiguous x -> add x ct
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      (* The solve with op(A) transposes to the solve with op(A)ᵀ, the conjugate
         of the other solve applied to the conjugate cotangent. *)
      add b
        (Nx.conjugate
           (eval
              (Solve_triangular
                 {
                   upper;
                   transpose = not transpose;
                   unit_diag;
                   a;
                   b = Nx.conjugate ct;
                 })))
  | Move (x, m) -> (
      match[@warning "@4@8"] m with
      | Reshape _ ->
          let ct = if Nx.is_c_contiguous ct then ct else Nx.contiguous ct in
          add x (Nx.reshape (Nx.shape x) ct)
      | Expand _ -> add x (unbroadcast ct (Nx.shape x))
      | Permute p ->
          let inverse = Array.make (Array.length p) 0 in
          Array.iteri (fun i j -> inverse.(j) <- i) p;
          add x (Nx.transpose ~axes:(Array.to_list inverse) ct)
      | Shrink limits ->
          let shape = Nx.shape x in
          add x
            (Nx.pad
               (Array.mapi (fun i (lo, hi) -> (lo, shape.(i) - hi)) limits)
               (Nx_dtype.zero (Nx.dtype x))
               ct)
      | Flip dims ->
          let axes =
            List.filter
              (fun i -> dims.(i))
              (List.init (Array.length dims) Fun.id)
          in
          add x (Nx.flip ~axes ct)
      | Window { axis; size; step } ->
          (* Overlap-add: input position [w * step + j] receives the cotangent
             of window [w] at offset [j], which is what fold sums. *)
          let r = Nx.ndim x in
          let folded =
            eval
              (Fold
                 {
                   output_size = [| (Nx.shape x).(axis) |];
                   kernel_size = [| size |];
                   stride = [| step |];
                   dilation = [| 1 |];
                   padding = [| (0, 0) |];
                   x = Nx.moveaxis axis r ct;
                 })
          in
          add x (Nx.moveaxis (r - 1) axis folded))
  | Matmul (a, b) ->
      if owns a then
        add a (unbroadcast (Nx.matmul ct (Nx.matrix_transpose b)) (Nx.shape a))
      else
        add b (unbroadcast (Nx.matmul (Nx.matrix_transpose a) ct) (Nx.shape b))
  | Place (_, x) -> add x (Nx.place (Nx.placement x) ct)
  | Compare _ | Sort _ | Threefry _ | Cholesky _ | Arg_reduce _ | Argsort _
  | Read _ ->
      assert false (* Never recorded. *)

let transpose_call cts i inputs like pullback =
  let received = ref false in
  let outputs =
    List.mapi
      (fun j (Nx.P y) ->
        match cts.cts.(i + j) with
        | Some ct ->
            received := true;
            ct
        | None -> Nx.P (Nx.zeros_like y))
      like
  in
  if !received then
    List.iter2
      (fun (Nx.P x) ct -> add cts x (Nx.unpack (Nx.dtype x) ct))
      inputs (pullback outputs)

let transpose cts =
  let t = cts.tape in
  for i = t.length - 1 downto 0 do
    match t.entries.(i) with
    | Recorded op ->
        Option.iter
          (fun ct -> transpose_op cts op (Nx.unpack (Nx.Op.dtype op) ct))
          cts.cts.(i)
    | Call { inputs; like; pullback } ->
        transpose_call cts i inputs like pullback
    | Input | Part -> ()
  done

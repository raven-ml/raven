(* Columns that hold lists, tensors, records and extension values. *)

open Talon

let show q = Format.printf "%a@.@." Talon.pp (Error.get_ok (Query.run q))

(* An extension type: lengths in metres, stored as float64. The stored order is
   the order of the lengths, so the declaration is ordered. *)
let metres =
  Ext.v ~name:"example.metres" ~ordered:true Type.float64 ~dec:Fun.id
    ~enc:Fun.id

let ext_type = Type.ext ~name:"example.metres" Type.float64

let () =
  (* A list column is offsets into one flat buffer, which nx reads in place. *)
  let tokens =
    Column.v (Type.list Type.int32) [| [| 5; 17; 2 |]; [||]; [| 9; 9 |] |]
  in
  let ragged = Column.ragged Nx.int32 tokens in
  Format.printf "lengths %a@.values %a@.@." Nx.pp (Nx_ragged.lengths ragged)
    Nx.pp (Nx_ragged.values ragged);

  (* A tensor column holds one tensor of a fixed shape per row. *)
  let images =
    Column.of_tensor
      (Nx.reshape [| 3; 2; 2 |] (Nx.arange_f Nx.float32 0. 12. 1.))
  in

  (* A record column holds named fields per row, each a value or null. *)
  let point x y =
    Record.(empty |> add Kind.float "x" (Some x) |> add Kind.float "y" y)
  in
  let points =
    Column.v
      Type.(record [ ("x", Any float64); ("y", Any float64) ])
      [| point 1. (Some 2.); point 3. None; point 0.5 (Some 0.5) |]
  in

  (* An extension column is laid out as its storage. *)
  let lengths =
    Column.of_layout (Any ext_type)
      (Column.layout (Column.v Type.float64 [| 1.5; 0.2; 12. |]))
    |> Result.get_ok
  in
  let t =
    Talon.v
      [
        ("id", Column.v Type.int8 [| 1; 2; 3 |]);
        ("tokens", tokens);
        ("image", images);
        ("point", points);
        ("length", lengths);
      ]
  in
  Format.printf "%a@.@." Schema.pp (Talon.schema t);
  show (Query.of_table t);

  (* Only the declaration reads an extension column's values. *)
  let length = Ext.col metres "length" in
  let long = Query.(of_table t |> filter Expr.(length > const 1.)) in
  let ids = Error.get_ok (Query.values (Col.int "id") long) in
  let ms = Error.get_ok (Query.values length long) in
  Array.iter2 (fun id m -> Format.printf "id %d: %g m@." id m) ids ms;

  (* Tensor and list cells read as OCaml values too. *)
  let image = Column.values (Kind.tensor Nx.float32) images in
  Format.printf "@.image 2: %a@." Nx.pp image.(1);
  Column.options (Kind.list Kind.int) tokens
  |> Array.iter (function
    | Some l ->
        Format.printf "[%s]@."
          (String.concat "; " (Array.to_list (Array.map string_of_int l)))
    | None -> Format.printf "null@.")

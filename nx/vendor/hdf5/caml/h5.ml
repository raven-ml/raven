open Bigarray
open Hdf5_raw

let escape s =
  let i = try String.index s '\\' with Not_found -> -1 in
  let j = try String.index s '/'  with Not_found -> -1 in
  if i = -1 && j = -1 then s
  else begin
    let len = String.length s in
    let buf = Buffer.create (len + 16) in
    for i = 0 to len - 1 do
      let c = String.get s i in
      if c == '\\' then begin
        Buffer.add_char buf '\\';
        Buffer.add_char buf '\\'
      end else if c == '/' then begin
        Buffer.add_char buf '\\';
        Buffer.add_char buf '-'
      end else
        Buffer.add_char buf c
    done;
    Buffer.contents buf
  end

let string_contains s c =
  let i = ref 0 in
  let len = String.length s in
  while !i < len && String.unsafe_get s !i <> c do
    incr i
  done;
  !i < len

let unescape s =
  if string_contains s '\\' then begin
    let len = String.length s in
    let pos = ref 0 in
    let buf = Buffer.create len in
    while !pos < len do
      let c = String.get s !pos in
      if c == '\\' then begin
        incr pos;
        let c = String.get s !pos in
        Buffer.add_char buf (if c = '-' then '/' else c)
      end else
        Buffer.add_char buf c;
      incr pos
    done;
    Buffer.contents buf
  end else s

let () = H5_raw.init ()

let default_get_set init =
  let ref = ref init in
  (fun () -> !ref), (fun v -> ref := v)

let default_deflate, set_default_deflate =
  default_get_set 6

let default_split, set_default_split = default_get_set false

type t =
| Dataset of Hid.t
| File of Hid.t
| Group of Hid.t

let hid = function
| Dataset d -> d
| File f -> f
| Group g -> g

let dataset d =
  match H5i.get_type d with
  | H5i.Type.DATASET -> File d
  | _ -> raise H5i.Fail

let file f =
  match H5i.get_type f with
  | H5i.Type.FILE -> File f
  | _ -> raise H5i.Fail

let group g =
  match H5i.get_type g with
  | H5i.Type.GROUP -> Group g
  | _ -> raise H5i.Fail

let h5t x =
  let open H5i in
  let open Type in
  match get_type x with
  | FILE -> File x
  | GROUP-> Group x
  | DATASET-> Dataset x
  | _ -> raise Fail

type open_ = ?meta_block_size:int -> ?split:bool -> string -> t

let open_ (open_ : string -> ?fapl : Hid.t -> H5f.Acc.t list -> Hid.t) acc =
  fun ?meta_block_size ?(split = default_split ()) name ->
    let fapl = H5p.create FILE_ACCESS in
    begin
      match meta_block_size with
      | None -> ()
      | Some meta_block_size -> H5p.set_meta_block_size fapl meta_block_size
    end;
    if split then begin
      H5p.set_fapl_split fapl "-m.h5" "-r.h5";
      (* Warm up metadata in the cache *)
      if Sys.file_exists (name ^ "-m.h5") then
        let command = "cat " ^ name ^ "-m.h5 > /dev/null &" in
        if Sys.command command <> 0 then
          failwith (Printf.sprintf "Error executing command \"%s\"" command)
    end;
    let result = open_ name ~fapl acc in
    H5p.close fapl;
    File result

let create_trunc : open_ =
  open_ (fun name ?fapl accs -> H5f.create name ?fapl accs) H5f.Acc.([ TRUNC ])

let open_rdonly = open_ H5f.open_ H5f.Acc.([ RDONLY ])

let open_rdwr = open_ H5f.open_ H5f.Acc.([ CREAT; RDWR ])

let open_group t name =
  let name = escape name in
  let t = hid t in
  Group (if name = "." || H5l.exists t name then H5g.open_ t name else H5g.create t name)

let open_dataset t name =
  Dataset (H5d.open_ (hid t) (escape name))

let close = function
| Dataset d -> H5d.close d
| File f -> H5f.close f
| Group g -> H5g.close g

let with_group t name f =
  let t = open_group t name in
  let r = f t in
  close t;
  r

let flush t = H5f.flush (hid t) H5f.Scope.LOCAL

let get_name t = H5f.get_name (hid t) |> unescape

let exists t name = H5l.exists (hid t) (escape name)

let delete t name = H5l.delete (hid t) (escape name)

let ls ?(index = H5_raw.Index.NAME) ?(order = H5_raw.Iter_order.NATIVE) t =
  let links = ref [] in
  let _ = H5l.iterate (hid t) index order (fun _ l _ () ->
    links := unescape l :: !links;
    H5_raw.Iter.CONT) () in
  List.rev !links

let copy ~src ~src_name ~dst ~dst_name =
  H5o.copy (hid src) (escape src_name) (hid dst) (escape dst_name)

let rec merge ~src ~dst ~on_duplicate path =
  let _ = H5l.iterate src H5_raw.Index.NAME H5_raw.Iter_order.NATIVE
    (fun _ name _ () ->
      if H5l.exists dst name then begin
        let src = H5o.open_ src name in
        let dst = H5o.open_ dst name in
        begin match H5i.get_type src, H5i.get_type dst with
        | t, t' when t <> t' ->
          invalid_arg (Printf.sprintf
            "Object %s not of the same type in source and destination" name)
        | H5i.Type.FILE, _
        | H5i.Type.GROUP, _ -> merge ~src ~dst ~on_duplicate (name :: path)
        | H5i.Type.DATASET, _ ->
          begin
            match on_duplicate with
            | `Skip -> ()
            | `Raise ->
              let b = Buffer.create 64 in
              Buffer.add_string b "H5.merge: duplicate dataset at ";
              List.iter (fun s ->
                Buffer.add_string b s;
                Buffer.add_string b "/") (List.rev path);
              Buffer.add_string b name;
              failwith (Buffer.contents b)
            | `Call f -> f (List.rev (name :: path))
          end
        | H5i.Type.DATATYPE, _
        | H5i.Type.DATASPACE, _
        | H5i.Type.ATTR, _
        (* | H5i.Type.REFERENCE removed in HDF5 1.14 *)
        | H5i.Type.VFL, _
        | H5i.Type.GENPROP_CLS, _
        | H5i.Type.GENPROP_LST, _
        | H5i.Type.ERROR_CLASS, _
        | H5i.Type.ERROR_MSG, _
        | H5i.Type.ERROR_STACK, _
        | H5i.Type.NTYPES, _ ->
          invalid_arg (Printf.sprintf "Unhandled object type %d for the object %s"
            (Obj.magic (H5i.get_type src)) name)
        end;
        H5o.close src;
        H5o.close dst
      end else H5o.copy src name dst name;
      H5_raw.Iter.CONT
    ) () in ()

let merge ~src ~dst ~on_duplicate = merge ~src:(hid src) ~dst:(hid dst) ~on_duplicate []

let create_hard_link ~obj ~obj_name ~link ~link_name =
  H5l.create_hard (hid obj) (escape obj_name) (hid link) (escape link_name)

let create_soft_link ~target_path ~link ~link_name =
  H5l.create_soft target_path (hid link) (escape link_name)

let create_external_link t ~target_file_name ~target_obj_name ~link_name =
  H5l.create_external target_file_name (escape target_obj_name) (hid t) (escape link_name)

let write_data write t datatype dims name ?(deflate = 6) data =
  let dataspace = H5s.create_simple dims in
  let dcpl =
    match deflate with
    | 0 -> None
    | deflate ->
      let dcpl = H5p.create H5p.Cls_id.DATASET_CREATE in
      H5p.set_chunk dcpl dims;
      H5p.set_deflate dcpl deflate;
      Some dcpl
  in
  let dataset = H5d.create (hid t) (escape name) datatype ?dcpl dataspace in
  write dataset datatype H5s.all H5s.all ?xfer_plist:None data;
  H5d.close dataset;
  H5s.close dataspace;
  match dcpl with
  | None -> ()
  | Some dcpl -> H5p.close dcpl

let read_data read expected_datatype create verify t data ?xfer_plist name =
  let hid = hid t in
  let dataset = H5d.open_ hid (escape name) in
  let datatype = H5d.get_type dataset in
  if not (H5t.equal expected_datatype datatype) then
    invalid_arg "Unexpected datatype";
  let dataspace = H5d.get_space dataset in
  let dims, _ = H5s.get_simple_extent_dims dataspace in
  let data =
    match data with
    | Some data -> verify data dims
    | None -> create dims in
  read dataset datatype H5s.all H5s.all ?xfer_plist data;
  H5s.close dataspace;
  H5t.close datatype;
  H5d.close dataset;
  data

let write_uint8_array1 t name ?deflate (a : (char, int8_unsigned_elt, _) Array1.t) =
  write_data H5d.write_bigarray t H5t.native_b8 [| Array1.dim a |] name ?deflate
    (genarray_of_array1 a)

let write_string t name ?deflate (a : string) =
  write_data H5d.write_string t H5t.c_s1 [| String.length a |] name ?deflate a

let write_string_array t name ?deflate (a : string array) =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype H5t.variable;
  write_data H5d.write_string_array t datatype [| Array.length a |] name ?deflate a;
  H5t.close datatype

module type Float_arg = sig
  type float_elt

  val h5t : Hid.t

  val kind : (float, float_elt) kind
end

module Make_float(F : Float_arg) = struct
  let write_float_array t name ?deflate (a : float array) =
    write_data H5d.write_float_array t F.h5t [| Array.length a |] name ?deflate a

  let write_float_genarray t name ?deflate (a : (float, F.float_elt, _) Genarray.t) =
    let a = Genarray.change_layout a C_layout in
    write_data H5d.write_bigarray t F.h5t (Genarray.dims a) name ?deflate a

  let write_float_array1 t name ?deflate (a : (float, F.float_elt, _) Array1.t) =
    write_float_genarray t name ?deflate (genarray_of_array1 a)

  let write_float_array2 t name ?deflate (a : (float, F.float_elt, _) Array2.t) =
    write_float_genarray t name ?deflate (genarray_of_array2 a)

  let write_float_array3 t name ?deflate (a : (float, F.float_elt, _) Array3.t) =
    write_float_genarray t name ?deflate (genarray_of_array3 a)

  let read_float_genarray t ?data name layout =
    let a = read_data H5d.read_bigarray F.h5t
      (fun dims -> Genarray.create F.kind c_layout dims)
      (fun data dims ->
        let data = Genarray.change_layout data C_layout in
        let dims' = Genarray.dims data in
        let n  = Array.length dims in
        let n' = Array.length dims' in
        let smaller = ref false in
        for i = 0 to min n n' - 1 do
          if dims'.(i) < dims.(i) then smaller := true
        done;
        if n <> n' || !smaller then
          invalid_arg (Printf.sprintf
            "The provided storage (%s) not of adequate size and dimensions (%s)"
            (Array.map string_of_int dims' |> Array.to_list |> String.concat " ")
            (Array.map string_of_int dims  |> Array.to_list |> String.concat " "));
        data)
      t data name in
    Genarray.change_layout a layout

  let read_float_array t ?data name =
    read_data H5d.read_float_array F.h5t
      (fun dims ->
        if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
        Array.create_float dims.(0))
      (fun data dims ->
        if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
        if Array.length data < dims.(0) then
          invalid_arg "The provided data storage too small";
        data)
      t data name

  let read_float_array1 t ?data name layout =
    read_float_genarray t name layout
      ?data:(match data with None -> None | Some data -> Some (genarray_of_array1 data))
    |> array1_of_genarray

  let read_float_array2 t ?data name layout =
    read_float_genarray t name layout
      ?data:(match data with None -> None | Some data -> Some (genarray_of_array2 data))
    |> array2_of_genarray

  let read_float_array3 t ?data name layout =
    read_float_genarray t name layout
      ?data:(match data with None -> None | Some data -> Some (genarray_of_array3 data))
    |> array3_of_genarray

  let write_attribute_float t name v =
    let dataspace = H5s.create H5s.Class.SCALAR in
    let att = H5a.create (hid t) name F.h5t dataspace in
    H5a.write_float att F.h5t v;
    H5a.close att;
    H5s.close dataspace

  let read_attribute_float t name =
    let att = H5a.open_ (hid t) name in
    let dataspace = H5a.get_space att in
    let datatype = H5a.get_type att in
    let f = H5a.read_float att datatype in
    H5t.close datatype;
    H5s.close dataspace;
    H5a.close att;
    f

  let write_attribute_float_array t name v =
    let dataspace = H5s.create_simple [| Array.length v |] in
    let att = H5a.create (hid t) name F.h5t dataspace in
    H5a.write_float_array att F.h5t v;
    H5a.close att;
    H5s.close dataspace

  let read_attribute_float_array t name =
    let att = H5a.open_ (hid t) name in
    let dataspace = H5a.get_space att in
    let datatype = H5a.get_type att in
    let dims, _ = H5s.get_simple_extent_dims dataspace in
    let a = Array.create_float dims.(0) in
    H5a.read_float_array att datatype a;
    H5t.close datatype;
    H5s.close dataspace;
    H5a.close att;
    a
end

module Float32 = Make_float(struct
  type float_elt = float32_elt
  let h5t = H5t.native_float
  let kind = Float32
end)

module Float64 = Make_float(struct
  type float_elt = float64_elt
  let h5t = H5t.native_double
  let kind = Float64
end)

include Float64

let read_uint8_array1 t ?data name layout =
  read_data H5d.read_bigarray H5t.native_b8
    (fun dims ->
      if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
      genarray_of_array1 (Array1.create Char layout dims.(0)))
    (fun data dims ->
      if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
      if Array1.dim data < dims.(0) then
        invalid_arg "The provided data storage too small";
      genarray_of_array1 data)
    t data name
  |> array1_of_genarray

let read_string t name =
  read_data H5d.read_string H5t.c_s1
    (fun dims ->
      if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
      Bytes.create dims.(0) |> Bytes.to_string)
    (fun _ _ -> assert false) t None name

let read_string_array t name =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype H5t.variable;

  let data = read_data H5d.read_c_string_array datatype
    (fun dims ->
      if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
      Array.make dims.(0) C_string.null)
    (fun _ _ -> assert false) t None name in
  H5t.close datatype;
  Array.map (fun cs ->
    let s = C_string.to_string cs in
    C_string.free cs;
    s) data

let read_bigstring_array t name =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype H5t.variable;

  let data = read_data H5d.read_c_string_array datatype
    (fun dims ->
      if Array.length dims <> 1 then invalid_arg "Dataset not one dimensional";
      Array.make dims.(0) C_string.null)
    (fun _ _ -> assert false) t None name in
  H5t.close datatype;
  (* We will reuse [data] *)
  for i = 0 to Array.length data - 1 do
    Array.unsafe_get data i
    |> C_string.to_bigstring
    |> Array.unsafe_set (Obj.magic data) i
  done;
  Obj.magic data

let write_attribute_int64 t name v =
  let dataspace = H5s.create H5s.Class.SCALAR in
  let att = H5a.create (hid t) name H5t.native_int64 dataspace in
  H5a.write_int64 att H5t.native_int64 v;
  H5a.close att;
  H5s.close dataspace

let read_attribute_int64 t name =
  let att = H5a.open_ (hid t) name in
  let dataspace = H5a.get_space att in
  let datatype = H5a.get_type att in
  let i = H5a.read_int64 att datatype in
  H5t.close datatype;
  H5s.close dataspace;
  H5a.close att;
  i

let write_attribute_string t name v =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype (String.length v);
  let dataspace = H5s.create H5s.Class.SCALAR in
  let att = H5a.create (hid t) name datatype dataspace in
  H5a.write_string att datatype v;
  H5a.close att;
  H5s.close dataspace;
  H5t.close datatype

let read_attribute_string t name =
  let att = H5a.open_ (hid t) name in
  let dataspace = H5a.get_space att in
  let datatype = H5a.get_type att in
  let a = Bytes.create (H5t.get_size datatype) in
  H5a.read_string att datatype a;
  H5t.close datatype;
  H5s.close dataspace;
  H5a.close att;
  Bytes.unsafe_to_string a

let write_attribute_string_array t name a =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype H5t.variable;
  let dataspace = H5s.create_simple [| Array.length a |] in
  let att = H5a.create (hid t) name datatype dataspace in
  H5a.write_string_array att datatype a;
  H5a.close att;
  H5s.close dataspace;
  H5t.close datatype

let read_attribute_string_array t name =
  let att = H5a.open_ (hid t) name in
  let dataspace = H5a.get_space att in
  let datatype = H5a.get_type att in
  let dims, _ = H5s.get_simple_extent_dims dataspace in
  let a = Array.make dims.(0) "" in
  H5a.read_string_array att datatype a;
  H5t.close datatype;
  H5s.close dataspace;
  H5a.close att;
  a

let write_attribute_c_string t name v =
  let datatype = H5t.copy H5t.c_s1 in
  H5t.set_size datatype H5t.variable;
  let dataspace = H5s.create H5s.Class.SCALAR in
  let att = H5a.create (hid t) name datatype dataspace in
  H5a.write_c_string att datatype v;
  H5a.close att;
  H5s.close dataspace;
  H5t.close datatype

let read_attribute_c_string t name =
  let att = H5a.open_ (hid t) name in
  let dataspace = H5a.get_space att in
  let datatype = H5a.get_type att in
  let a = H5a.read_c_string att datatype in
  H5t.close datatype;
  H5s.close dataspace;
  H5a.close att;
  a

let attribute_exists t name = H5a.exists (hid t) name

let delete_attribute t name = H5a.delete (hid t) name

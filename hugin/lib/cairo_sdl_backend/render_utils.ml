let set_source_color cr (color : Artist.color) =
  Cairo.set_source_rgba cr color.r color.g color.b color.a

let to_cairo_surface ?cmap (data : Nx.uint8_t) =
  let shape = Nx.shape data in
  let h, w, channels =
    match shape with
    | [| h; w; 4 |] -> (h, w, 4)
    | [| h; w; 3 |] -> (h, w, 3)
    | [| h; w |] -> (h, w, 1)
    | _ -> failwith "Render_utils.to_cairo_surface: Unsupported image shape"
  in
  if h <= 0 || w <= 0 then
    failwith "Render_utils.to_cairo_surface: Invalid image dimensions";

  let stride = Cairo.Image.stride_for_width Cairo.Image.ARGB32 w in
  let cairo_data =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (stride * h)
  in
  Bigarray.Array1.fill cairo_data 0;

  let cairo_surface =
    Cairo.Image.create_for_data8 cairo_data Cairo.Image.ARGB32 ~w ~h ~stride
  in

  let module BA = Bigarray in
  let src_ba =
    let ba = Nx.to_bigarray data in
    let size = Array.fold_left ( * ) 1 (Nx.dims data) in
    BA.reshape_1 ba size
  in

  let () =
    match channels with
    | 4 ->
        let src_stride = w * 4 in
        for r = 0 to h - 1 do
          let src_offset = r * src_stride in
          let dst_offset = r * stride in
          for c = 0 to w - 1 do
            let r_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 4) + 0)
            in
            let g_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 4) + 1)
            in
            let b_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 4) + 2)
            in
            let a_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 4) + 3)
            in
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 0) b_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 1) g_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 2) r_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 3) a_val
          done
        done
    | 3 ->
        let src_stride = w * 3 in
        for r = 0 to h - 1 do
          let src_offset = r * src_stride in
          let dst_offset = r * stride in
          for c = 0 to w - 1 do
            let r_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 3) + 0)
            in
            let g_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 3) + 1)
            in
            let b_val =
              BA.Array1.unsafe_get src_ba (src_offset + (c * 3) + 2)
            in
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 0) b_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 1) g_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 2) r_val;
            BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 3) 255
          done
        done
    | 1 -> (
        match cmap with
        | None ->
            let src_stride = w in
            for r = 0 to h - 1 do
              let src_offset = r * src_stride in
              let dst_offset = r * stride in
              for c = 0 to w - 1 do
                let v = BA.Array1.unsafe_get src_ba (src_offset + c) in
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 0) v;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 1) v;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 2) v;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 3) 255
              done
            done
        | Some cmap_val ->
            let src_stride = w in
            for r = 0 to h - 1 do
              let src_offset = r * src_stride in
              let dst_offset = r * stride in
              for c = 0 to w - 1 do
                let v_u8 = BA.Array1.unsafe_get src_ba (src_offset + c) in
                let v_norm = float_of_int v_u8 /. 255.0 in
                let r_cm, g_cm, b_cm =
                  Artist.Colormap.apply_colormap cmap_val v_norm
                in
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 0) b_cm;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 1) g_cm;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 2) r_cm;
                BA.Array1.unsafe_set cairo_data (dst_offset + (c * 4) + 3) 255
              done
            done)
    | _ -> failwith "Render_utils.to_cairo_surface: Unexpected channel count"
  in
  cairo_surface

let float32_to_cairo_surface ?(cmap = Artist.Colormap.gray)
    (data : Nx.float32_t) =
  let shape = Nx.shape data in
  let h, w, channels =
    match shape with
    | [| h; w |] -> (h, w, 1)
    | [| h; w; 3 |] -> (h, w, 3)
    | [| h; w; 4 |] -> (h, w, 4)
    | _ ->
        failwith
          "Render_utils.float32_to_cairo_surface: Unsupported image shape"
  in
  if h <= 0 || w <= 0 then
    failwith "Render_utils.float32_to_cairo_surface: Invalid image dimensions";

  let stride = Cairo.Image.stride_for_width Cairo.Image.ARGB32 w in
  let cairo_data =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (stride * h)
  in
  Bigarray.Array1.fill cairo_data 0;

  let module BA = Bigarray in
  let src_ba : (float, BA.float32_elt, BA.c_layout) BA.Genarray.t =
    Nx.to_bigarray data
  in

  let clamp01 f = max 0.0 (min 1.0 f) in
  let to_u8 f = max 0 (min 255 (int_of_float ((f *. 255.) +. 0.5))) in

  let () =
    match channels with
    | 1 ->
        let src_a =
          (Obj.magic src_ba : (float, BA.float32_elt, BA.c_layout) BA.Array2.t)
        in
        let min_v = ref Float.max_float in
        let max_v = ref Float.min_float in
        for r = 0 to h - 1 do
          for c = 0 to w - 1 do
            let v = BA.Array2.get src_a r c in
            if Float.is_finite v then (
              if v < !min_v then min_v := v;
              if v > !max_v then max_v := v)
          done
        done;
        let range = !max_v -. !min_v in
        let norm_factor = if range > 1e-9 then 1.0 /. range else 0.0 in

        for r = 0 to h - 1 do
          let dst_offset_base = r * stride in
          for c = 0 to w - 1 do
            let v = BA.Array2.get src_a r c in
            let normalized_v =
              if Float.is_finite v then clamp01 ((v -. !min_v) *. norm_factor)
              else 0.0
            in
            let r_u8, g_u8, b_u8 =
              Artist.Colormap.apply_colormap cmap normalized_v
            in
            let dst_offset = dst_offset_base + (c * 4) in
            BA.Array1.unsafe_set cairo_data (dst_offset + 0) b_u8;
            BA.Array1.unsafe_set cairo_data (dst_offset + 1) g_u8;
            BA.Array1.unsafe_set cairo_data (dst_offset + 2) r_u8;
            BA.Array1.unsafe_set cairo_data (dst_offset + 3) 255
          done
        done
    | 3 ->
        let src_a =
          (Obj.magic src_ba : (float, BA.float32_elt, BA.c_layout) BA.Array3.t)
        in
        for r = 0 to h - 1 do
          let dst_offset_base = r * stride in
          for c = 0 to w - 1 do
            let r_f = clamp01 (BA.Array3.get src_a r c 0) in
            let g_f = clamp01 (BA.Array3.get src_a r c 1) in
            let b_f = clamp01 (BA.Array3.get src_a r c 2) in
            let dst_offset = dst_offset_base + (c * 4) in
            BA.Array1.unsafe_set cairo_data (dst_offset + 0) (to_u8 b_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 1) (to_u8 g_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 2) (to_u8 r_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 3) 255
          done
        done
    | 4 ->
        let src_a =
          (Obj.magic src_ba : (float, BA.float32_elt, BA.c_layout) BA.Array3.t)
        in
        for r = 0 to h - 1 do
          let dst_offset_base = r * stride in
          for c = 0 to w - 1 do
            let r_f = clamp01 (BA.Array3.get src_a r c 0) in
            let g_f = clamp01 (BA.Array3.get src_a r c 1) in
            let b_f = clamp01 (BA.Array3.get src_a r c 2) in
            let a_f = clamp01 (BA.Array3.get src_a r c 3) in
            let dst_offset = dst_offset_base + (c * 4) in
            BA.Array1.unsafe_set cairo_data (dst_offset + 0) (to_u8 b_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 1) (to_u8 g_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 2) (to_u8 r_f);
            BA.Array1.unsafe_set cairo_data (dst_offset + 3) (to_u8 a_f)
          done
        done
    | _ ->
        failwith
          "Render_utils.float32_to_cairo_surface: Unexpected channel count \
           (internal error)"
  in

  Cairo.Image.create_for_data8 cairo_data Cairo.Image.ARGB32 ~w ~h ~stride

type rgb8 =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let clamp_byte value =
  let clipped = if value < 0. then 0. else if value > 1. then 1. else value in
  int_of_float (Float.round (clipped *. 255.))

let create_rgb8 ~width ~height =
  let len = width * height * 3 in
  let buffer : rgb8 =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout len
  in
  let image =
    let open Fehu.Render in
    {
      width;
      height;
      pixel_format = `RGB8;
      data_u8 = Some buffer;
      data_f32 = None;
    }
  in
  (image, buffer)

let copy_u8 data =
  let len = Bigarray.Array1.dim data in
  let buffer : rgb8 =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout len
  in
  Bigarray.Array1.blit data buffer;
  buffer

let convert_rgba8_to_rgb8 data =
  let src_len = Bigarray.Array1.dim data in
  if src_len mod 4 <> 0 then
    invalid_arg "convert_rgba8_to_rgb8: stride mismatch";
  let pixels = src_len / 4 in
  let buffer : rgb8 =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (pixels * 3)
  in
  let rec loop idx_src idx_dst remaining =
    if remaining = 0 then ()
    else
      let r = Bigarray.Array1.unsafe_get data idx_src in
      let g = Bigarray.Array1.unsafe_get data (idx_src + 1) in
      let b = Bigarray.Array1.unsafe_get data (idx_src + 2) in
      Bigarray.Array1.unsafe_set buffer idx_dst r;
      Bigarray.Array1.unsafe_set buffer (idx_dst + 1) g;
      Bigarray.Array1.unsafe_set buffer (idx_dst + 2) b;
      loop (idx_src + 4) (idx_dst + 3) (remaining - 1)
  in
  loop 0 0 pixels;
  buffer

let convert_gray8_to_rgb8 data =
  let len = Bigarray.Array1.dim data in
  let buffer : rgb8 =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (len * 3)
  in
  let rec loop idx_src idx_dst remaining =
    if remaining = 0 then ()
    else
      let v = Bigarray.Array1.unsafe_get data idx_src in
      Bigarray.Array1.unsafe_set buffer idx_dst v;
      Bigarray.Array1.unsafe_set buffer (idx_dst + 1) v;
      Bigarray.Array1.unsafe_set buffer (idx_dst + 2) v;
      loop (idx_src + 1) (idx_dst + 3) (remaining - 1)
  in
  loop 0 0 len;
  buffer

let convert_float_to_rgb8 data channels =
  let len = Bigarray.Array1.dim data in
  if len mod channels <> 0 then
    invalid_arg "convert_float_to_rgb8: stride mismatch";
  let pixels = len / channels in
  let buffer : rgb8 =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (pixels * 3)
  in
  let rec loop idx_src idx_dst remaining =
    if remaining = 0 then ()
    else
      let r = clamp_byte (Bigarray.Array1.unsafe_get data idx_src) in
      let g = clamp_byte (Bigarray.Array1.unsafe_get data (idx_src + 1)) in
      let b_index = if channels = 3 then idx_src + 2 else idx_src + 2 in
      let b = clamp_byte (Bigarray.Array1.unsafe_get data b_index) in
      Bigarray.Array1.unsafe_set buffer idx_dst (Char.unsafe_chr r);
      Bigarray.Array1.unsafe_set buffer (idx_dst + 1) (Char.unsafe_chr g);
      Bigarray.Array1.unsafe_set buffer (idx_dst + 2) (Char.unsafe_chr b);
      loop (idx_src + channels) (idx_dst + 3) (remaining - 1)
  in
  loop 0 0 pixels;
  buffer

let convert_to_rgb8 src_image =
  let open Fehu.Render in
  match (src_image.pixel_format, src_image.data_u8, src_image.data_f32) with
  | `RGB8, Some data, _ ->
      let buffer = copy_u8 data in
      ({ src_image with data_u8 = Some buffer; data_f32 = None }, buffer)
  | `RGBA8, Some data, _ ->
      let buffer = convert_rgba8_to_rgb8 data in
      let image =
        let open Fehu.Render in
        {
          width = src_image.width;
          height = src_image.height;
          pixel_format = `RGB8;
          data_u8 = Some buffer;
          data_f32 = None;
        }
      in
      (image, buffer)
  | `GRAY8, Some data, _ ->
      let buffer = convert_gray8_to_rgb8 data in
      let image =
        let open Fehu.Render in
        {
          width = src_image.width;
          height = src_image.height;
          pixel_format = `RGB8;
          data_u8 = Some buffer;
          data_f32 = None;
        }
      in
      (image, buffer)
  | `RGBf, _, Some data ->
      let buffer = convert_float_to_rgb8 data 3 in
      let image =
        let open Fehu.Render in
        {
          width = src_image.width;
          height = src_image.height;
          pixel_format = `RGB8;
          data_u8 = Some buffer;
          data_f32 = None;
        }
      in
      (image, buffer)
  | `RGBAf, _, Some data ->
      let buffer = convert_float_to_rgb8 data 4 in
      let image =
        let open Fehu.Render in
        {
          width = src_image.width;
          height = src_image.height;
          pixel_format = `RGB8;
          data_u8 = Some buffer;
          data_f32 = None;
        }
      in
      (image, buffer)
  | _ -> invalid_arg "convert_to_rgb8: unsupported image payload"

let rgb24_bytes_of_image image =
  let _, buffer = convert_to_rgb8 image in
  let len = Bigarray.Array1.dim buffer in
  let bytes = Bytes.create len in
  for idx = 0 to len - 1 do
    Bytes.unsafe_set bytes idx (Bigarray.Array1.unsafe_get buffer idx)
  done;
  bytes

let set_pixel_rgb8 ~buffer ~width ~x ~y ~r ~g ~b =
  if x < 0 || y < 0 then ()
  else
    let height = Bigarray.Array1.dim buffer / (width * 3) in
    if x >= width || y >= height then ()
    else
      let base = ((y * width) + x) * 3 in
      Bigarray.Array1.unsafe_set buffer base (Char.unsafe_chr r);
      Bigarray.Array1.unsafe_set buffer (base + 1) (Char.unsafe_chr g);
      Bigarray.Array1.unsafe_set buffer (base + 2) (Char.unsafe_chr b)

let fill_rect_rgb8 ~buffer ~width ~x ~y ~w ~h ~r ~g ~b =
  for row = 0 to h - 1 do
    for col = 0 to w - 1 do
      set_pixel_rgb8 ~buffer ~width ~x:(x + col) ~y:(y + row) ~r ~g ~b
    done
  done

let blit_rgb8 ~src ~src_width ~src_height ~dst ~dst_width ~x ~y =
  for row = 0 to src_height - 1 do
    let src_base = row * src_width * 3 in
    let dst_base = (((y + row) * dst_width) + x) * 3 in
    for col = 0 to (src_width * 3) - 1 do
      let value = Bigarray.Array1.unsafe_get src (src_base + col) in
      Bigarray.Array1.unsafe_set dst (dst_base + col) value
    done
  done

let compose_grid ~rows ~cols frames =
  if rows <= 0 || cols <= 0 then
    invalid_arg "compose_grid: rows and cols must be positive";
  let expected = rows * cols in
  if Array.length frames <> expected then
    invalid_arg "compose_grid: frame count mismatch with grid layout";
  let first_image, first_buffer = convert_to_rgb8 frames.(0) in
  let frame_width = first_image.width in
  let frame_height = first_image.height in
  let grid_image, grid_buffer =
    create_rgb8 ~width:(frame_width * cols) ~height:(frame_height * rows)
  in
  (* Place the first frame. *)
  blit_rgb8 ~src:first_buffer ~src_width:frame_width ~src_height:frame_height
    ~dst:grid_buffer ~dst_width:(frame_width * cols) ~x:0 ~y:0;
  Array.iteri
    (fun idx frame ->
      if idx = 0 then ()
      else
        let image_rgb, buffer = convert_to_rgb8 frame in
        if image_rgb.width <> frame_width || image_rgb.height <> frame_height
        then invalid_arg "compose_grid: all frames must share dimensions";
        let row = idx / cols in
        let col = idx mod cols in
        blit_rgb8 ~src:buffer ~src_width:frame_width ~src_height:frame_height
          ~dst:grid_buffer ~dst_width:(frame_width * cols)
          ~x:(col * frame_width) ~y:(row * frame_height))
    frames;
  grid_image

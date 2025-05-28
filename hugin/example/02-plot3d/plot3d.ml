open Nx

let create_helix_data () =
  let t = linspace float32 0. (4. *. Float.pi) 100 in
  let x = cos t in
  let y = sin t in
  let z = map_item (fun t -> t /. (4. *. Float.pi)) t in
  (x, y, z)

let () =
  let x, y, z = create_helix_data () in
  let fig =
    Hugin.plot3d ~title:"3D Helix" ~xlabel:"x" ~ylabel:"y" ~zlabel:"z" x y z
  in
  Hugin.show fig

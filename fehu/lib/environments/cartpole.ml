open Fehu

type observation = (float, Rune.float32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = string

type state = {
  mutable x : float; (* cart position *)
  mutable x_dot : float; (* cart velocity *)
  mutable theta : float; (* pole angle *)
  mutable theta_dot : float; (* pole angular velocity *)
  mutable steps : int;
  rng : Rune.Rng.key ref;
}

(* Environment parameters matching Gymnasium CartPole-v1 *)
let gravity = 9.8
let masscart = 1.0
let masspole = 0.1
let total_mass = masscart +. masspole
let length = 0.5 (* half the pole's length *)
let polemass_length = masspole *. length
let force_mag = 10.0
let tau = 0.02 (* time step *)

(* Thresholds for episode termination *)
let theta_threshold_radians = 12. *. Float.pi /. 180.
let x_threshold = 2.4

let observation_space =
  Space.Box.create
    ~low:
      [|
        -4.8;
        -.Float.max_float;
        -.theta_threshold_radians *. 2.;
        -.Float.max_float;
      |]
    ~high:
      [| 4.8; Float.max_float; theta_threshold_radians *. 2.; Float.max_float |]

let action_space = Space.Discrete.create 2

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "ansi"
  |> Metadata.with_description (Some "Classic cart-pole balancing problem")
  |> Metadata.add_author "Fehu"
  |> Metadata.with_version (Some "0.1.0")

let reset _env ?options:_ () state =
  (* Reset to small random values around 0 *)
  let keys = Rune.Rng.split !(state.rng) ~n:5 in
  state.rng := keys.(0);

  (* Uniform random values in [-0.05, 0.05] *)
  let random_state i =
    let r = Rune.rand Rune.float32 ~key:keys.(i + 1) [| 1 |] in
    let v = (Rune.to_array r).(0) in
    (v -. 0.5) *. 0.1
  in

  state.x <- random_state 0;
  state.x_dot <- random_state 1;
  state.theta <- random_state 2;
  state.theta_dot <- random_state 3;
  state.steps <- 0;

  let obs =
    Rune.create Rune.float32 [| 4 |]
      [| state.x; state.x_dot; state.theta; state.theta_dot |]
  in
  (obs, Info.empty)

let step _env action state =
  let action_value =
    let arr : Int32.t array = Rune.to_array action in
    Int32.to_int arr.(0)
  in

  let force = if action_value = 1 then force_mag else -.force_mag in

  let costheta = cos state.theta in
  let sintheta = sin state.theta in

  (* Equations from Gymnasium CartPole-v1 *)
  let temp =
    (force
    +. (polemass_length *. state.theta_dot *. state.theta_dot *. sintheta))
    /. total_mass
  in
  let thetaacc =
    ((gravity *. sintheta) -. (costheta *. temp))
    /. (length
       *. ((4.0 /. 3.0) -. (masspole *. costheta *. costheta /. total_mass)))
  in
  let xacc = temp -. (polemass_length *. thetaacc *. costheta /. total_mass) in

  (* Euler integration *)
  state.x <- state.x +. (tau *. state.x_dot);
  state.x_dot <- state.x_dot +. (tau *. xacc);
  state.theta <- state.theta +. (tau *. state.theta_dot);
  state.theta_dot <- state.theta_dot +. (tau *. thetaacc);
  state.steps <- state.steps + 1;

  let terminated =
    state.x < -.x_threshold || state.x > x_threshold
    || state.theta < -.theta_threshold_radians
    || state.theta > theta_threshold_radians
  in

  let truncated = state.steps >= 500 in
  let reward = if terminated then 0.0 else 1.0 in

  let obs =
    Rune.create Rune.float32 [| 4 |]
      [| state.x; state.x_dot; state.theta; state.theta_dot |]
  in

  let info = Info.set "steps" (Info.int state.steps) Info.empty in
  Env.transition ~observation:obs ~reward ~terminated ~truncated ~info ()

let render state =
  Printf.sprintf
    "CartPole: x=%.3f, x_dot=%.3f, theta=%.3f°, theta_dot=%.3f, steps=%d"
    state.x state.x_dot
    (state.theta *. 180. /. Float.pi)
    state.theta_dot state.steps

let make ~rng () =
  let state =
    {
      x = 0.0;
      x_dot = 0.0;
      theta = 0.0;
      theta_dot = 0.0;
      steps = 0;
      rng = ref rng;
    }
  in
  Env.create ~id:"CartPole-v1" ~metadata ~rng ~observation_space ~action_space
    ~reset:(fun env ?options () -> reset env ?options () state)
    ~step:(fun env action -> step env action state)
    ~render:(fun _ -> Some (render state))
    ~close:(fun _ -> ())
    ()

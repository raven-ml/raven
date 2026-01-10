let load_mnist_like_dataset ~fashion_mnist =
  let (x_train_ba, y_train_ba), (x_test_ba, y_test_ba) =
    Mnist.load ~fashion_mnist
  in
  let x_train = Nx.of_buffer (Nx_buffer.genarray_of_array3 x_train_ba) in
  let y_train = Nx.of_buffer (Nx_buffer.genarray_of_array1 y_train_ba) in
  let x_test = Nx.of_buffer (Nx_buffer.genarray_of_array3 x_test_ba) in
  let y_test = Nx.of_buffer (Nx_buffer.genarray_of_array1 y_test_ba) in

  let train_count = Nx_buffer.Array3.dim1 x_train_ba in
  let test_count = Nx_buffer.Array3.dim1 x_test_ba in
  let height = Nx_buffer.Array3.dim2 x_train_ba in
  let width = Nx_buffer.Array3.dim3 x_train_ba in

  let x_train = Nx.reshape [| train_count; height; width; 1 |] x_train in
  let x_test = Nx.reshape [| test_count; height; width; 1 |] x_test in
  let y_train = Nx.reshape [| train_count; 1 |] y_train in
  let y_test = Nx.reshape [| test_count; 1 |] y_test in
  ((x_train, y_train), (x_test, y_test))

let load_mnist () = load_mnist_like_dataset ~fashion_mnist:false
let load_fashion_mnist () = load_mnist_like_dataset ~fashion_mnist:true

let load_cifar10 () =
  let (x_train_ba, y_train_ba), (x_test_ba, y_test_ba) = Cifar10.load () in
  let x_train = Nx.of_buffer x_train_ba in
  let y_train = Nx.of_buffer (Nx_buffer.genarray_of_array1 y_train_ba) in
  let x_test = Nx.of_buffer x_test_ba in
  let y_test = Nx.of_buffer (Nx_buffer.genarray_of_array1 y_test_ba) in

  let train_count = Nx_buffer.Array1.dim y_train_ba in
  let test_count = Nx_buffer.Array1.dim y_test_ba in
  let y_train = Nx.reshape [| train_count; 1 |] y_train in
  let y_test = Nx.reshape [| test_count; 1 |] y_test in
  ((x_train, y_train), (x_test, y_test))

let load_tabular_dataset loader_func =
  let features_ba, labels_ba = loader_func () in
  let features = Nx.of_buffer (Nx_buffer.genarray_of_array2 features_ba) in
  let labels = Nx.of_buffer (Nx_buffer.genarray_of_array1 labels_ba) in
  let num_samples = Nx_buffer.Array1.dim labels_ba in
  let labels = Nx.reshape [| num_samples; 1 |] labels in
  (features, labels)

let load_iris () = load_tabular_dataset Iris.load
let load_breast_cancer () = load_tabular_dataset Breast_cancer.load
let load_diabetes () = load_tabular_dataset Diabetes.load
let load_california_housing () = load_tabular_dataset California_housing.load

let load_time_series_dataset loader_func =
  let series_ba = loader_func () in
  Nx.of_buffer (Nx_buffer.genarray_of_array1 series_ba)

let load_airline_passengers () =
  load_time_series_dataset Airline_passengers.load

(* Include generators inline *)
include Generators

let get_cache_dir ?getenv dataset_name =
  Dataset_utils.get_cache_dir ?getenv dataset_name

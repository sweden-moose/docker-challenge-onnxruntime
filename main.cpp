// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>


void run_ort_cuda() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto& api = Ort::GetApi();
  
  // Создаем настройки сессии 
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Включаем basic и extended оптимизацию графа

  // Создаем настройки для провайдера CUDA 
  OrtCUDAProviderOptions options;
  options.device_id = 0;
  options.arena_extend_strategy = 0;
  options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Берем алгоритм для поиска сверток из примера
  options.do_copy_in_default_stream = 1;

#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

  // Подлючаем нашего CUDA-провайдера к сессии
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA(session_options, &options));

  std::cout << "Running ORT CUDA EP with default provider options" << std::endl;
  
  // Создаем сессию с подключенной моделью squeezenet.onnx v1.0-9 :)
  Ort::Session session(env, model_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Определяем количество входов модели
  const size_t num_input_nodes = session.GetInputCount();
  
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}. 
                                         // Otherwise need vector<vector<>>  <- Этот парень прав, раз вход один, все поместится в один вектор

  std::cout << "Number of inputs = " << num_input_nodes << std::endl;

  // Проходимся по каждому входной узлу
  for (size_t i = 0; i < num_input_nodes; i++) {
    // Выводим имя узла
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

    // Выводим тип узла
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input " << i << " : type = " << type << std::endl;

    // Выводим оазмерность входного тензора
    input_node_dims = tensor_info.GetShape();
    std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Input " << i << " : dim[" << j << "] =" << input_node_dims[j] << '\n';
    }
    std::cout << std::flush;
  }

  constexpr size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size <- Этот парень прав, эти параметры модели неизменны, можно и вхардкодить.
                                                       // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // Создание тензора из входных данных
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                            input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // Выполняем модель с использованием входных данных, получаем результат в виде выходного тензора.
  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Берем указатель на результаты выходного тензора
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // Выводим вероятности для первых пяти классов
  for (int i = 0; i < 5; i++) {
    std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
  }
  std::cout << std::flush;

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317

  std::cout << "Done!" << std::endl;
}

int main(int /*argc*/, char*[]) {
  run_ort_cuda();
  return 0;
}

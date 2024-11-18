#include <c10/cuda/CUDAException.h>

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <chrono>
#include <string>

namespace c10 {

static int device = -1;
static bool recording = false;
static int64_t sync_time, sync_start;
static int64_t sync_stream, sync_device, sync_event;

void SyncRecorder::init_record()
{
    if (device == -1) {
        C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    }
    if (device) return;
    recording = true;
    sync_time = 0;
    sync_stream = 0;
    sync_device = 0;
    sync_event = 0;
}

int64_t SyncRecorder::finish_record()
{
    if (device) return 0;
    recording = false;
    // std::cout << "sync_stream :" << sync_stream << " times." << std::endl;
    // std::cout << "sync_device :" << sync_device << " times." << std::endl;
    // std::cout << "sync_event :" << sync_event << " times." << std::endl;
    return sync_time;
}

void SyncRecorder::start_record(int type)
{
    if (device || !recording) return;
    auto dur = std::chrono::high_resolution_clock::now().time_since_epoch();
    sync_start = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
    // if (type == 1) {
    //     sync_stream++;
    // } else if (type == 2) {
    //     sync_device++;
    // } else if (type == 3) {
    //     sync_event++;
    // }
}

void SyncRecorder::end_record()
{
    if (device || !recording) return;
    auto dur = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto sync_end = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
    sync_time += sync_end - sync_start;
}

namespace cuda {

void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  const auto cuda_error = static_cast<cudaError_t>(err);
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  auto error_unused C10_UNUSED = cudaGetLastError();
  (void)error_unused;

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA error: ");
  check_message.append(cudaGetErrorString(cuda_error));
  check_message.append(c10::cuda::get_cuda_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif

  TORCH_CHECK(false, check_message);
}

} // namespace cuda
} // namespace c10

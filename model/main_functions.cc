/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model/model_data.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "uFire_SHT20.h"
#include "hx_drv_tflm.h"
#include "hx_example_utils.h"
// Globals, used for compatibility with Arduino-style sketches.
uFire_SHT20 mySensor;

namespace
{
    hx_drv_gpio_config_t gpio_config;
}

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 399 * 1024;
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined (__CCAC__)
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".tensor_arena")));
#else
#pragma Bss(".tensor_arena")
static uint8_t tensor_arena[kTensorArenaSize];
#pragma Bss()
#endif // if defined (_GNUC_) && !defined (_CCAC_)
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(final_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSplit();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  
  //Sensor
  hx_drv_share_switch(SHARE_MODE_I2CM);
  hx_drv_uart_initial(UART_BR_115200);
  //hx_drv_led_on(HX_DRV_LED_GREEN);
  //hx_drv_led_on(HX_DRV_LED_RED);

  gpio_config.gpio_pin = HX_DRV_PGPIO_0;
  gpio_config.gpio_direction = HX_DRV_GPIO_OUTPUT;
  hx_drv_gpio_initial(&gpio_config);
  gpio_config.gpio_pin = HX_DRV_PGPIO_1;
  gpio_config.gpio_direction = HX_DRV_GPIO_INPUT;
  hx_drv_gpio_initial(&gpio_config);
  gpio_config.gpio_pin = HX_DRV_PGPIO_2;
  gpio_config.gpio_direction = HX_DRV_GPIO_OUTPUT;
  hx_drv_gpio_initial(&gpio_config);

  hx_drv_uart_print("SHT20\n"); 
  hx_drv_uart_print("US100\n");      
// Init SHT20 Sensor    
  mySensor.begin();    
                    
  if (mySensor.begin() == false)
  {
    hx_drv_uart_print("SHT20 error. Please check wiring. Freezing...\n");
  }
  hx_util_delay_ms(100);
}

unsigned long pingMotor() 
{
  uint32_t i = 0, n = 300;
  while(i<=n)
  {
    gpio_config.gpio_pin = HX_DRV_PGPIO_2;
    gpio_config.gpio_data = 1;
    hx_drv_gpio_set(&gpio_config);
    hx_util_delay_ms(1);

    gpio_config.gpio_pin = HX_DRV_PGPIO_2;
    gpio_config.gpio_data = 0;
    hx_drv_gpio_set(&gpio_config);
    hx_util_delay_ms(19);
    hx_drv_uart_print("motor %d\n",i);      
    i +=1;
  }
  return 0;

}

unsigned long ping() {
    uint32_t tick_start = 0, tick_end = 0;
    gpio_config.gpio_pin = HX_DRV_PGPIO_0;
    gpio_config.gpio_data = 1;
    hx_drv_gpio_set(&gpio_config);
    hx_util_delay_ms(5);

    gpio_config.gpio_pin = HX_DRV_PGPIO_0;
    gpio_config.gpio_data = 0;
    hx_drv_gpio_set(&gpio_config);

    uint32_t time = 0;
    uint32_t i = 0;

    while(1)
    {
        gpio_config.gpio_pin = HX_DRV_PGPIO_1;
        hx_drv_gpio_get(&gpio_config);
        time = gpio_config.gpio_data;
        //hx_drv_uart_print(":%d",time);
        
        //hx_drv_uart_print("%d\n",i);
        if(time == 1 && i == 0)
            {
                hx_drv_tick_start();
                hx_drv_tick_get(&tick_start);
            }
        if (time == 1)
            {
                i++;
            }
        if(time == 0 && i > 0)
            {
                hx_drv_tick_get(&tick_end);
                break;
            }
    }
    uint32_t cm = ((tick_end-tick_start)/4)*0.0001657;
    //hx_drv_uart_print("%d",cm);
    return  (cm) ;  // 換算成 cm 並傳回
  }

unsigned long getCatImage() 
{
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t cat_score = output->data.uint8[kPersonIndex];
  int8_t no_cat_score = output->data.uint8[kNotAPersonIndex]; 
  uint32_t cat = RespondToDetection(error_reporter, cat_score, no_cat_score,(int8_t*)output->data.uint8); 
    return cat;
}
// The name of this function is important for Arduino compatibility.
void loop() {
      // Read Humidity Read Temperature 
  hx_drv_uart_print("Humidity: %d %%,  Temperature: %d C\n",(uint32_t)mySensor.humidity(),(uint32_t)mySensor.temperature());

  uint32_t distance  = ping();  
  hx_drv_uart_print("Distance: %d cm\n", distance);
  hx_drv_uart_print("\n"); 
  hx_util_delay_ms(500);
  pingMotor();
  if(distance<40){
      while (true) {
        uint32_t tick_start = 0, tick_end = 0;
        uint32_t isCat = getCatImage(); 
        hx_drv_uart_print("is cat ? :%d \n", isCat);
        distance  = ping();  
        if(distance>40){
          break;
        }
      }
  }
}

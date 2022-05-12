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

#if defined(ARDUINO)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO)

#ifndef ARDUINO_EXCLUDE_CODE

#include "model/model_settings.h"
#include "model/detection_responder.h"
#include "hx_drv_tflm.h"  // NOLINT

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.
namespace {
uint8_t score_output[kCategoryCount];
}

unsigned long RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t cat_score, int8_t no_cat_score, int8_t* score) {
  uint32_t cat;
  if (cat_score > no_cat_score) {
    hx_drv_led_on(HX_DRV_LED_GREEN);
    cat = 0;
  } else {
    hx_drv_led_off(HX_DRV_LED_GREEN);
    cat = 1;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "cat score:%d not cat score %d",
                       cat_score, no_cat_score);
  score_output[0] = score[0] + 128;
  score_output[1] = score[1] + 128;
   //send result data out through SPI
  hx_drv_spim_send((uint32_t)score_output, sizeof(int8_t) * kCategoryCount,
                   SPI_TYPE_META_DATA);
    return cat;
}

#endif  // ARDUINO_EXCLUDE_CODE

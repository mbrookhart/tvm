/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file resize.cc
 * \brief Image resize operators
 */
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace dyn {

TVM_REGISTER_NODE_TYPE(ResizeAttrs);

bool ResizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // {data, size, out}
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const ResizeAttrs* param = attrs.as<ResizeAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "Resize only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, Any());
  oshape.Set(3, Any());

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // assign output type
  reporter->Assign(types[2], TensorType(layout_converter.BackwardShape(oshape), out_dtype));
  return true;
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize(Expr data, Expr size, String layout, String method,
                String coordinate_transformation_mode, String rounding_method, double bicubic_alpha,
                double bicubic_exclude, DataType out_dtype) {
  auto attrs = make_object<ResizeAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->bicubic_alpha = bicubic_alpha;
  attrs->bicubic_exclude = bicubic_exclude;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("dyn.image.resize");
  return Call(op, {data, size}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.image._make.resize").set_body_typed(MakeResize);

RELAY_REGISTER_OP("dyn.image.resize")
    .describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **size**: data is 2D array of shape (2,) with values
            (new_height, new_width)

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ResizeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("size", "Tensor", "The output size tensor.")
    .set_support_level(5)
    .add_type_rel("DynResize", ResizeRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

bool CropAndResizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* boxes = types[1].as<TensorTypeNode>();
  const auto* box_indices = types[2].as<TensorTypeNode>();
  if (data == nullptr || boxes == nullptr || box_indices == nullptr) return false;

  const CropAndResizeAttrs* param = attrs.as<CropAndResizeAttrs>();
  ICHECK(param != nullptr);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // 4-D tensor of shape [num_boxes, crop_height, crop_width, depth]
  static const Layout kNCHW("NCHW");
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(0, boxes->shape[0]);
  oshape.Set(2, Any());
  oshape.Set(3, Any());
  auto bshape = layout_converter.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[4], TensorType(bshape, out_dtype));
  return true;
}

Expr MakeCropAndResize(Expr data, Expr boxes, Expr box_indices, Expr crop_size, String layout,
                       String method, double extrapolation_value, DataType out_dtype) {
  auto attrs = make_object<CropAndResizeAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->extrapolation_value = std::move(extrapolation_value);
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("dyn.image.crop_and_resize");
  return Call(op, {data, boxes, box_indices, crop_size}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.image._make.crop_and_resize").set_body_typed(MakeCropAndResize);

RELAY_REGISTER_OP("dyn.image.crop_and_resize")
    .describe(
        R"code(Perform crop and resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, crop_size[0], crop_size[1])

           for layout NHWC
           (batch_size, crop_size[0], crop_size[1], channels)
)code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("boxes", "Tensor", "The boxes tensor.")
    .add_argument("box_indices", "Tensor", "The box indices tensor.")
    .add_argument("size", "Tensor", "The box indices tensor.")
    .set_attrs_type<CropAndResizeAttrs>()
    .set_support_level(5)
    .add_type_rel("DynCropAndResize", CropAndResizeRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_utils.hpp"

#include <cugraph-ops/dimenet/agg_edge_to_edge.hpp>
#include <cugraph-ops/dimenet/radial_basis.hpp>

#include <pybind11/pybind11.h>

#include <torch/torch.h>

#include <tuple>
#include <vector>

namespace torch_extension {

static constexpr int NUM_RADIAL = 6;
static constexpr int NUM_SPHERICAL = 7;

constexpr int mma_prec2int(cugraph::ops::MMAOpT mma_prec) {
  return static_cast<int>(mma_prec);
}

inline cugraph::ops::MMAOpT int2mma_prec(int v) {
  // TODO(mjoux) this assumes that SIMT will always be the last element
  // that should be fine for tests, but ultimately, we want to add a
  // `Count` element or something to help with this.
  TORCH_CHECK(v >= 0 && v <= mma_prec2int(cugraph::ops::MMAOpT::kSimt));
  return static_cast<cugraph::ops::MMAOpT>(v);
}

using torch::autograd::AutogradContext;

class RadialBasisBackward : public torch::autograd::Function<RadialBasisBackward> {
public:
    static std::vector<at::Tensor> forward(
      AutogradContext *ctx,
      at::Tensor grad_rbf_i,
      at::Tensor grad_sbf_rad_i,
      at::Tensor x,
      at::Tensor w,
      bool needs_grad_x,
      bool needs_grad_w)
    {
      auto n = int32_t(x.size(0));
      auto grad_rbf = grad_rbf_i.contiguous();
      auto grad_sbf_rad = grad_sbf_rad_i.contiguous();
      ctx->save_for_backward({grad_rbf, grad_sbf_rad, x, w});

      at::optional<at::Tensor> grad_x;
      at::optional<at::Tensor> grad_w;

      if (needs_grad_x)
        grad_x = torch::empty_like(x);
      if (needs_grad_w)
        grad_w = torch::zeros_like(w);

      auto stream = c10::cuda::getCurrentCUDAStream();
      TORCH_CHECK(
        x.scalar_type() == w.scalar_type() &&
        x.scalar_type() == grad_rbf.scalar_type() &&
        x.scalar_type() == grad_sbf_rad.scalar_type(),
        "RBF: all inputs must have same type, got ", x.scalar_type(),
        ", ", w.scalar_type(), ", ", grad_rbf.scalar_type(), ", ", grad_sbf_rad.scalar_type()
      );
      if (x.scalar_type() == c10::ScalarType::Float) {
        cugraph::ops::dimenet::radial_basis_bwd(
          maybe_ptr<float>(grad_x),
          maybe_ptr<float>(grad_w),
          get_ptr<float>(grad_rbf),
          get_ptr<float>(grad_sbf_rad),
          get_ptr<float>(x),
          get_ptr<float>(w),
          n,
          stream);
      } else if (x.scalar_type() == c10::ScalarType::Double) {
        cugraph::ops::dimenet::radial_basis_bwd(
          maybe_ptr<double>(grad_x),
          maybe_ptr<double>(grad_w),
          get_ptr<double>(grad_rbf),
          get_ptr<double>(grad_sbf_rad),
          get_ptr<double>(x),
          get_ptr<double>(w),
          n,
          stream);
      } else {
        TORCH_CHECK(false, "RBF: unsupported type ", x.scalar_type());
      }
      return {grad_x.value_or(at::Tensor()), grad_w.value_or(at::Tensor())};
    }

    static std::vector<at::Tensor> backward(
      AutogradContext *ctx,
      std::vector<at::Tensor> grad_grad_i)
    {
      auto saved = ctx->get_saved_variables();
      auto grad_grad_x = grad_grad_i[0].contiguous();
      auto grad_grad_w = grad_grad_i[1].contiguous();

      auto stream = c10::cuda::getCurrentCUDAStream();
      auto grad_grad_rbf = torch::empty_like(saved[0]);
      auto grad_grad_sbf_rad = torch::empty_like(saved[1]);
      auto grad_w = torch::zeros_like(saved[3]);
      auto n = int32_t(saved[2].size(0));

      TORCH_CHECK(
        grad_grad_x.scalar_type() == grad_grad_w.scalar_type() &&
        grad_grad_x.scalar_type() == saved[0].scalar_type(),
        "RBF: all inputs must have same type, got ", grad_grad_x.scalar_type(),
        ", ", grad_grad_w.scalar_type(), ", ", saved[0].scalar_type()
      );
      if (grad_grad_x.scalar_type() == c10::ScalarType::Float) {
        cugraph::ops::dimenet::radial_basis_bwd_bwd(
          get_ptr<float>(grad_grad_rbf),
          get_ptr<float>(grad_grad_sbf_rad),
          get_ptr<float>(grad_w),
          get_ptr<float>(grad_grad_x),
          get_ptr<float>(grad_grad_w),
          get_ptr<float>(saved[0]),
          get_ptr<float>(saved[1]),
          get_ptr<float>(saved[2]),
          get_ptr<float>(saved[3]),
          n,
          stream);
      } else if (grad_grad_x.scalar_type() == c10::ScalarType::Double) {
        cugraph::ops::dimenet::radial_basis_bwd_bwd(
          get_ptr<double>(grad_grad_rbf),
          get_ptr<double>(grad_grad_sbf_rad),
          get_ptr<double>(grad_w),
          get_ptr<double>(grad_grad_x),
          get_ptr<double>(grad_grad_w),
          get_ptr<double>(saved[0]),
          get_ptr<double>(saved[1]),
          get_ptr<double>(saved[2]),
          get_ptr<double>(saved[3]),
          n,
          stream);
      } else {
        TORCH_CHECK(false, "RBF: unsupported type ", grad_grad_x.scalar_type());
      }
      return {grad_grad_rbf, grad_grad_sbf_rad, at::Tensor(), grad_w, at::Tensor(), at::Tensor()};
    }
};

class RadialBasis : public torch::autograd::Function<RadialBasis> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx, at::Tensor x_i, at::Tensor w_i) {
    // TODO(mjoux): in later versions, can just use ctx->needs_input_grad(0/1) directly
    ctx->saved_data["needs_grad_x"] = at::IValue(x_i.requires_grad());
    ctx->saved_data["needs_grad_w"] = at::IValue(w_i.requires_grad());
    auto x = x_i.contiguous();
    auto w = w_i.contiguous();
    auto stream = c10::cuda::getCurrentCUDAStream();
    ctx->save_for_backward({x, w});
    auto n = int32_t(x.size(0));
    auto rbf = torch::empty({n, NUM_RADIAL}, x.options());
    auto sbf_rad = torch::empty({n, NUM_RADIAL * NUM_SPHERICAL}, x.options());
    TORCH_CHECK(
      x.scalar_type() == w.scalar_type(),
      "RBF: both inputs must have same type, got ", x.scalar_type(),
      " and ", w.scalar_type()
    );
    if (x.scalar_type() == c10::ScalarType::Float) {
      cugraph::ops::dimenet::radial_basis_fwd(
        get_ptr<float>(rbf),
        get_ptr<float>(sbf_rad),
        get_ptr<float>(x),
        get_ptr<float>(w),
        n,
        stream);
    } else if (x.scalar_type() == c10::ScalarType::Double) {
      cugraph::ops::dimenet::radial_basis_fwd(
        get_ptr<double>(rbf),
        get_ptr<double>(sbf_rad),
        get_ptr<double>(x),
        get_ptr<double>(w),
        n,
        stream);
    } else {
      TORCH_CHECK(false, "RBF: unsupported type ", x.scalar_type());
    }
    return {rbf, sbf_rad};
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    return RadialBasisBackward::apply(
      grad_outs[0], grad_outs[1], saved[0], saved[1],
      ctx->saved_data["needs_grad_x"].to<bool>(), ctx->saved_data["needs_grad_w"].to<bool>()
    );
  }
};

std::tuple<at::Tensor, at::Tensor> radial_basis_wrapped(
  const at::Tensor& x, const at::Tensor& w)
{
  auto result = RadialBasis::apply(x, w);
  return std::make_tuple(result[0], result[1]);
}

class IBAggEdgeBackward : public torch::autograd::Function<IBAggEdgeBackward> {
public:
    static std::vector<at::Tensor> forward(
      AutogradContext *ctx,
      at::Tensor grad_e_out_ft_i,
      at::Tensor e_vec,
      at::Tensor e_rbf,
      at::Tensor e_in_ft,
      at::Tensor sbf_weights,
      at::Tensor edge_index,
      at::Tensor src_offsets,
      at::Tensor dst_offsets,
      at::Tensor dst_e_idx,
      cugraph::ops::MMAOpT mma_precision)
    {
      TORCH_CHECK(!e_vec.requires_grad(),
        "ib_agg_edge: expected input position difference vector to not require "
        "gradient (be detached from graph) "
      );
      // TODO(mjoux): in later versions, can just use ctx->needs_input_grad(0) directly
      ctx->saved_data["needs_grad_out"] = at::IValue(grad_e_out_ft_i.requires_grad());
      ctx->saved_data["mma_prec"] = at::IValue(mma_prec2int(mma_precision));
      auto grad_e_out_ft = grad_e_out_ft_i.contiguous();
      ctx->save_for_backward({
        grad_e_out_ft, e_vec, e_rbf, e_in_ft, sbf_weights, edge_index,
        src_offsets, dst_offsets, dst_e_idx
      });

      auto stream = c10::cuda::getCurrentCUDAStream();
      
      auto grad_e_rbf = torch::empty_like(e_rbf);
      auto grad_e_in_ft = torch::empty_like(e_in_ft);
      auto grad_sbf_weights = torch::empty_like(sbf_weights);

      if (e_in_ft.scalar_type() == c10::ScalarType::Float) {
        cugraph::ops::dimenet::agg_edge_to_edge_bwd(
          get_ptr<float>(grad_e_rbf),
          get_ptr<float>(grad_e_in_ft),
          get_ptr<float>(grad_sbf_weights),
          edge_index.size(1),
          get_ptr<float>(e_vec),
          get_ptr<float>(e_rbf),
          get_ptr<float>(e_in_ft),
          get_ptr<float>(grad_e_out_ft),
          e_in_ft.size(1),
          get_ptr<float>(sbf_weights),
          sbf_weights.size(1),
          get_ptr<int64_t>(edge_index),
          get_ptr<int64_t>(src_offsets),
          mma_precision,
          stream);
      } else if (e_in_ft.scalar_type() == c10::ScalarType::Double) {
        cugraph::ops::dimenet::agg_edge_to_edge_bwd(
          get_ptr<double>(grad_e_rbf),
          get_ptr<double>(grad_e_in_ft),
          get_ptr<double>(grad_sbf_weights),
          edge_index.size(1),
          get_ptr<double>(e_vec),
          get_ptr<double>(e_rbf),
          get_ptr<double>(e_in_ft),
          get_ptr<double>(grad_e_out_ft),
          e_in_ft.size(1),
          get_ptr<double>(sbf_weights),
          sbf_weights.size(1),
          get_ptr<int64_t>(edge_index),
          get_ptr<int64_t>(src_offsets),
          mma_precision,
          stream);
      } else {
        TORCH_CHECK(false, "ib_agg_edge: unsupported type ", e_in_ft.scalar_type());
      }
      return {grad_e_rbf, grad_e_in_ft, grad_sbf_weights};
    }

    static std::vector<at::Tensor> backward(
      AutogradContext *ctx,
      std::vector<at::Tensor> grad_grad_i)
    {
      auto saved = ctx->get_saved_variables();
      auto grad_grad_e_rbf = grad_grad_i[0].contiguous();
      auto grad_grad_e_in_ft = grad_grad_i[1].contiguous();
      auto grad_grad_sbf_weights = grad_grad_i[2].contiguous();

      auto stream = c10::cuda::getCurrentCUDAStream();
      auto grad_rbf_grad_e_in_ft = torch::empty_like(saved[2]);
      auto grad_e_in_ft_grad_e_rbf = torch::empty_like(saved[3]);
      auto grad_sbf_weights = torch::zeros_like(saved[4]);
      auto mma_prec = int2mma_prec(ctx->saved_data["mma_prec"].to<int>());

      if (saved[3].scalar_type() == c10::ScalarType::Float) {
        cugraph::ops::dimenet::agg_edge_to_edge_bwd2_main(
          get_ptr<float>(grad_rbf_grad_e_in_ft),
          get_ptr<float>(grad_e_in_ft_grad_e_rbf),
          get_ptr<float>(grad_sbf_weights),
          saved[5].size(1),
          get_ptr<float>(saved[1]),
          get_ptr<float>(saved[2]),
          get_ptr<float>(grad_grad_e_rbf),
          get_ptr<float>(saved[3]),
          get_ptr<float>(grad_grad_e_in_ft),
          get_ptr<float>(saved[0]),
          saved[3].size(1),
          get_ptr<float>(saved[4]),
          saved[4].size(1),
          get_ptr<int64_t>(saved[5]),
          get_ptr<int64_t>(saved[6]),
          mma_prec,
          stream);
      } else if (saved[3].scalar_type() == c10::ScalarType::Double) {
        cugraph::ops::dimenet::agg_edge_to_edge_bwd2_main(
          get_ptr<double>(grad_rbf_grad_e_in_ft),
          get_ptr<double>(grad_e_in_ft_grad_e_rbf),
          get_ptr<double>(grad_sbf_weights),
          saved[5].size(1),
          get_ptr<double>(saved[1]),
          get_ptr<double>(saved[2]),
          get_ptr<double>(grad_grad_e_rbf),
          get_ptr<double>(saved[3]),
          get_ptr<double>(grad_grad_e_in_ft),
          get_ptr<double>(saved[0]),
          saved[3].size(1),
          get_ptr<double>(saved[4]),
          saved[4].size(1),
          get_ptr<int64_t>(saved[5]),
          get_ptr<int64_t>(saved[6]),
          mma_prec,
          stream);
      } else {
        TORCH_CHECK(false, "ib_agg_edge: unsupported type ", saved[3].scalar_type());
      }

      at::optional<at::Tensor> grad_grad_e_out_ft;
      if (ctx->saved_data["needs_grad_out"].to<bool>()) {
        grad_grad_e_out_ft = torch::empty_like(saved[0]);
        if (saved[3].scalar_type() == c10::ScalarType::Float) {
          cugraph::ops::dimenet::agg_edge_to_edge_bwd2_grad(
            get_ptr<float>(grad_grad_e_out_ft.value()),
            saved[5].size(1),
            get_ptr<float>(saved[1]),
            get_ptr<float>(saved[2]),
            get_ptr<float>(grad_grad_e_rbf),
            get_ptr<float>(saved[3]),
            get_ptr<float>(grad_grad_e_in_ft),
            saved[3].size(1),
            get_ptr<float>(saved[4]),
            saved[4].size(1),
            get_ptr<int64_t>(saved[5]),
            get_ptr<int64_t>(saved[7]),
            get_ptr<int64_t>(saved[8]),
            mma_prec,
            stream);
        } else if (saved[3].scalar_type() == c10::ScalarType::Double) {
          cugraph::ops::dimenet::agg_edge_to_edge_bwd2_grad(
            get_ptr<double>(grad_grad_e_out_ft.value()),
            saved[5].size(1),
            get_ptr<double>(saved[1]),
            get_ptr<double>(saved[2]),
            get_ptr<double>(grad_grad_e_rbf),
            get_ptr<double>(saved[3]),
            get_ptr<double>(grad_grad_e_in_ft),
            saved[3].size(1),
            get_ptr<double>(saved[4]),
            saved[4].size(1),
            get_ptr<int64_t>(saved[5]),
            get_ptr<int64_t>(saved[7]),
            get_ptr<int64_t>(saved[8]),
            mma_prec,
            stream);
        } else {
          TORCH_CHECK(false, "ib_agg_edge: unsupported type ", saved[3].scalar_type());
        }
      }
      return {
        grad_grad_e_out_ft.value_or(at::Tensor()),
        at::Tensor(),
        grad_rbf_grad_e_in_ft,
        grad_e_in_ft_grad_e_rbf,
        grad_sbf_weights,
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()
      };
    }
};

class IBAggEdge : public torch::autograd::Function<IBAggEdge> {
public:
    static std::vector<at::Tensor> forward(
      AutogradContext *ctx,
      at::Tensor e_vec_i,
      at::Tensor e_rbf_i,
      at::Tensor e_in_ft_i,
      at::Tensor sbf_weights_i,
      at::Tensor edge_index,
      at::Tensor src_offsets,
      at::Tensor dst_offsets,
      at::Tensor dst_e_idx,
      cugraph::ops::MMAOpT mma_precision)
    {
      ctx->saved_data["mma_prec"] = at::IValue(mma_prec2int(mma_precision));
      auto e_vec = e_vec_i.contiguous();
      auto e_rbf = e_rbf_i.contiguous();
      auto e_in_ft = e_in_ft_i.contiguous();
      auto sbf_weights = sbf_weights_i.contiguous();
      auto stream = c10::cuda::getCurrentCUDAStream();
      ctx->save_for_backward({
        e_vec, e_rbf, e_in_ft, sbf_weights, edge_index,
        src_offsets, dst_offsets, dst_e_idx
      });

      auto e_out_ft = torch::empty_like(e_in_ft);

      if (e_in_ft.scalar_type() == c10::ScalarType::Float) {
        cugraph::ops::dimenet::agg_edge_to_edge_fwd(
          get_ptr<float>(e_out_ft),
          edge_index.size(1),
          get_ptr<float>(e_vec),
          get_ptr<float>(e_rbf),
          get_ptr<float>(e_in_ft),
          e_in_ft.size(1),
          get_ptr<float>(sbf_weights),
          sbf_weights.size(1),
          get_ptr<int64_t>(edge_index),
          get_ptr<int64_t>(dst_offsets),
          get_ptr<int64_t>(dst_e_idx),
          mma_precision,
          stream);
      } else if (e_in_ft.scalar_type() == c10::ScalarType::Double) {
        cugraph::ops::dimenet::agg_edge_to_edge_fwd(
          get_ptr<double>(e_out_ft),
          edge_index.size(1),
          get_ptr<double>(e_vec),
          get_ptr<double>(e_rbf),
          get_ptr<double>(e_in_ft),
          e_in_ft.size(1),
          get_ptr<double>(sbf_weights),
          sbf_weights.size(1),
          get_ptr<int64_t>(edge_index),
          get_ptr<int64_t>(dst_offsets),
          get_ptr<int64_t>(dst_e_idx),
          mma_precision,
          stream);
      } else {
        TORCH_CHECK(false, "ib_agg_edge: unsupported type ", e_in_ft.scalar_type());
      }
      return {e_out_ft};
    }

    static std::vector<at::Tensor> backward(
      AutogradContext *ctx,
      std::vector<at::Tensor> grad_outs)
    {
      auto mma_prec = int2mma_prec(ctx->saved_data["mma_prec"].to<int>());
      auto saved = ctx->get_saved_variables();
      auto bwd_result = IBAggEdgeBackward::apply(
        grad_outs[0], saved[0], saved[1], saved[2], saved[3], saved[4],
        saved[5], saved[6], saved[7], mma_prec
      );
      return {
        at::Tensor(),
        bwd_result[0],
        bwd_result[1],
        bwd_result[2],
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()
      };
    }
};

at::Tensor ib_agg_edge_wrapped(
  const at::Tensor& e_vec,
  const at::Tensor& e_rbf,
  const at::Tensor& e_in_ft,
  const at::Tensor& sbf_weights,
  const at::Tensor& edge_index,
  const at::Tensor& src_offsets,
  const at::Tensor& dst_offsets,
  const at::Tensor& dst_e_idx,
  cugraph::ops::MMAOpT mma_precision)
{
  return IBAggEdge::apply(
    e_vec, e_rbf, e_in_ft, sbf_weights, edge_index, src_offsets,
    dst_offsets, dst_e_idx, mma_precision
  )[0];
}

}  // namespace torch_extension

void init_torch_extension(pybind11::module_& m)
{
  pybind11::enum_<cugraph::ops::MMAOpT>(m, "MMAOp")
    .value("HighPrecision", cugraph::ops::MMAOpT::kHighPrecision)
    .value("LowPrecision", cugraph::ops::MMAOpT::kLowPrecision)
    .value("SIMT", cugraph::ops::MMAOpT::kSimt);

  m.def("radial_basis", &torch_extension::radial_basis_wrapped);
  m.def("ib_agg_edge", &torch_extension::ib_agg_edge_wrapped);
}

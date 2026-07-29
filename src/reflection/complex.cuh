#pragma once

#include <rayd/detail/field_math.h>

namespace rayd::torch_backend {

// Compatibility imports for existing Torch kernels. The implementation and
// POD layout are owned by the backend-neutral shared field contract.
using shared::field::Complex;
using shared::field::Complex3;
using shared::field::c_abs2;
using shared::field::c_add;
using shared::field::c_div;
using shared::field::c_exp_neg_i;
using shared::field::c_exp_neg_i_product;
using shared::field::c_make;
using shared::field::c_mul;
using shared::field::c_mul_real;
using shared::field::c_scale;
using shared::field::c_sqrt;
using shared::field::c_sub;
using shared::field::c3_add;
using shared::field::c3_dot_real;
using shared::field::c3_from_real;
using shared::field::c3_mul_complex;
using shared::field::c3_power;
using shared::field::c3_scale_complex;
using shared::field::c3_zero;
using shared::field::finite_complex3;

} // namespace rayd::torch_backend

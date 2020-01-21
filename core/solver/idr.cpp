/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/idr.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/idr_kernels.hpp"


namespace gko {
namespace solver {


namespace idr {


GKO_REGISTER_OPERATION(initialize, idr::initialize);
GKO_REGISTER_OPERATION(step_1, idr::step_1);
GKO_REGISTER_OPERATION(step_2, idr::step_2);


}  // namespace idr


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Vector = matrix::Dense<ValueType>;
    constexpr uint8 RelativeStoppingId{1};
    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    // TODO let user decide the shadow space number
    int s = 1;

    auto r = Vector::create_with_config_of(dense_b);
    auto r_norm = Vector::create(exec, dim<2>{1, r->get_size()[1]});        //?
    auto b_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});  //?

    auto P = Vector::create(exec, dim<2>{dense_b->get_size()[1],
                                         s});  // N = dense_b->get_size()[1] ?
    auto Dr = Vector::create(exec, dim<2>{dense_b->get_size()[1], s});
    auto Dx = Vector::create(exec, dim<2>{dense_b->get_size()[1], s});

    auto v = Vector::create_with_config_of(dense_b);  // A*r same size as b?
    double om = 0;
    auto M = Vector::create(
        exec, dim<2>{dense_b->get_size()[1], s});  // same size as P?
    int iter = -1;
    int oldest = 1;

    auto m = Vector::create_with_config_of(dense_b);  // P*r same size as b?
    auto c = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);
    auto dm = Vector::create_with_config_of(dense_b);

    auto tmp1 = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    ;
    auto tmp2 = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    ;


    // stopping criteria?
    bool one_changed{};
    // right exec?
    Array<stopping_status> stop_status(exec, dense_b->get_size()[1]);

    // TODO how should they be initialized exactly?(P random, aber
    // orthonormalbasis der zeilen)
    exec->run(idr::make_initialize(dense_b, r.get(), r_norm.get(), b_norm.get(),
                                   P.get(), Dr.get(), Dx.get(), v.get(),
                                   M.get(), m.get(), c.get(), q.get(), t.get(),
                                   dm.get(), &stop_status));

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          r.get());  // r = b-Ax

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    /*
    //check if solution is already sufficient
    if (stop_criterion->update()
            .num_iterations(iter)
            .residual(r.get())
            .solution(dense_x)
            .check(RelativeStoppingId, true, &stop_status, &one_changed))
            {
        break;
    }*/

    for (int k = 1; k < s; k++) {
        system_matrix_->apply(r.get(), v.get());

        v->compute_dot(r.get(), tmp1.get());
        v->compute_dot(v.get(), tmp2.get());
        exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                   &stop_status));
        // om = dot(v,r)/dot(v,v);
        // dX(:,k) = om*r; dR(:,k) = -om*v;
        // x = x + dX(:,k); r = r + dR(:,k);
        // normr = norm(r);
        // resvec = [resvec;normr];
        // M(:,k) = P*dR(:,k);
    }
    iter = s;
    P->apply(r.get(), m.get());
    while (true) {
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        for (int k = 0; k < s; k++) {
            exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                       &stop_status));  // step2
            // c = M\m; wie sollte man dieses system lösen?
            // q = -dR*c; % s-1 updates + 1 scaling
            // v = r + q;

            if (k == 0) {
                exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                           &stop_status));  // step3
                // t = A*v; % 1 matmul
                // om = dot(t,v)/dot(t,t); dot products wieder auslagern
                // dR(:,oldest) = q - om*t;
                // dX(:,oldest) = -dX*c + om*v;
            } else {
                exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                           &stop_status));  // step 4
                // dX(:,oldest) = -dX*c + om*v; % s updates + 1 scaling
                // dR(:,oldest) = -A*dX(:,oldest);
            }
            exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                       &stop_status));  // step 5
            // r = r + dR(:,oldest); % simple addition
            // x = x + dX(:,oldest);

            iter = iter + 1;

            exec->run(idr::make_step_1(q.get(), P.get(), Dr.get(), Dx.get(),
                                       &stop_status));  // step 6
            // normr=norm(r); % 1 inner product (not counted)
            // resvec = [resvec;normr];
            // dm = P*dR(:,oldest); % s inner products
            // M(:,oldest) = dm;
            // m = m + dm;


            oldest = oldest + 1;
            if (oldest > s) {
                oldest = 1;
            }
        }
    }


    // TODO (script:idr): change the code imported from solver/cg if needed
    //    using std::swap;
    //    using Vector = matrix::Dense<ValueType>;
    //
    //    constexpr uint8 RelativeStoppingId{1};
    //
    //    auto exec = this->get_executor();
    //
    //    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    //    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);
    //
    //    auto dense_b = as<const Vector>(b);
    //    auto dense_x = as<Vector>(x);
    //    auto r = Vector::create_with_config_of(dense_b);
    //    auto z = Vector::create_with_config_of(dense_b);
    //    auto p = Vector::create_with_config_of(dense_b);
    //    auto q = Vector::create_with_config_of(dense_b);
    //
    //    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    //    auto beta = Vector::create_with_config_of(alpha.get());
    //    auto prev_rho = Vector::create_with_config_of(alpha.get());
    //    auto rho = Vector::create_with_config_of(alpha.get());
    //
    //    bool one_changed{};
    //    Array<stopping_status> stop_status(alpha->get_executor(),
    //                                       dense_b->get_size()[1]);
    //
    //    // TODO: replace this with automatic merged kernel generator
    //    exec->run(idr::make_initialize(dense_b, r.get(), z.get(), p.get(),
    //    q.get(),
    //                                  prev_rho.get(), rho.get(),
    //                                  &stop_status));
    //    // r = dense_b
    //    // rho = 0.0
    //    // prev_rho = 1.0
    //    // z = p = q = 0
    //
    //    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
    //    r.get()); auto stop_criterion = stop_criterion_factory_->generate(
    //        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *)
    //        {}), x, r.get());
    //
    //    int iter = -1;
    //    while (true) {
    //        get_preconditioner()->apply(r.get(), z.get());
    //        r->compute_dot(z.get(), rho.get());
    //
    //        ++iter;
    //        this->template log<log::Logger::iteration_complete>(this, iter,
    //        r.get(),
    //                                                            dense_x);
    //        if (stop_criterion->update()
    //                .num_iterations(iter)
    //                .residual(r.get())
    //                .solution(dense_x)
    //                .check(RelativeStoppingId, true, &stop_status,
    //                &one_changed))
    //                {
    //            break;
    //        }
    //
    //        exec->run(idr::make_step_1(p.get(), z.get(), rho.get(),
    //        prev_rho.get(),
    //                                  &stop_status));
    //        // tmp = rho / prev_rho
    //        // p = z + tmp * p
    //        system_matrix_->apply(p.get(), q.get());
    //        p->compute_dot(q.get(), beta.get());
    //        exec->run(idr::make_step_2(dense_x, r.get(), p.get(), q.get(),
    //                                  beta.get(), rho.get(), &stop_status));
    //        // tmp = rho / beta
    //        // x = x + tmp * p
    //        // r = r - tmp * q
    //        swap(prev_rho, rho);
    //    }
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta,
                                LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:idr): change the code imported from solver/cg if needed
//    auto dense_x = as<matrix::Dense<ValueType>>(x);
//
//    auto x_clone = dense_x->clone();
//    this->apply(b, x_clone.get());
//    dense_x->scale(beta);
//    dense_x->add_scaled(alpha, x_clone.get());
//}


#define GKO_DECLARE_IDR(_type) class Idr<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR);


}  // namespace solver
}  // namespace gko

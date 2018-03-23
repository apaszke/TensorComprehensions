/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>

#include "pybind_utils.h"
#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/mapping_options.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/sema.h"
#include "tc/lang/tc_format.h"

namespace tc {
namespace python {

namespace py = pybind11;

using ATenCudaCompilationUnit = tc::ATenCompilationUnit<tc::CudaTcExecutor>;

static const lang::SourceRange dummyRange {std::make_shared<std::string>(""), 0, 0};

using namespace lang;

struct GradInfo {
  void addGradComprehension(Ident lhs_name,
                            ListView<lang::TreeRef> lhs_indices,
                            TreeRef rhs_expr) {
    int assignKind = has_grad_defined_.count(lhs_name.name()) > 0 ? TK_PLUS_EQ : TK_PLUS_EQ_B;
    grad_comps_.push_back(
      Comprehension::create(dummyRange,
                            lhs_name,
                            lhs_indices,
                            Compound::create(assignKind, dummyRange, {}),
                            rhs_expr,
                            ListView<TreeRef>::create(dummyRange, TreeList{}),
                            Compound::create(TK_OPTION, dummyRange, {}),
                            ListView<TreeRef>::create(dummyRange, TreeList{})));
    has_grad_defined_.insert(lhs_name.name());
  }
  bool has_zero_grad(const std::string& name) {
    return has_zero_grad_.count(name) > 0;
  }
  void mark_zero_grad(const std::string& name) {
    has_zero_grad_.count(name);
  }
  void mark_has_grad(const std::string& name) {
    has_grad_defined_.insert(name);
  }
  std::vector<lang::TreeRef>&& getGradComps() {
    return std::move(grad_comps_);
  }
private:
  std::vector<lang::TreeRef> grad_comps_;
  std::unordered_set<std::string> has_grad_defined_;
  std::unordered_set<std::string> has_zero_grad_;
};

void differentiateExpr(GradInfo& grad_info,
                       lang::TreeRef expr,
                       lang::TreeRef grad_output_expr) {
  using namespace lang;
  switch (expr->kind()) {
    case TK_ACCESS: {
      Access acc {expr};
      grad_info.addGradComprehension(acc.name(), acc.arguments(), grad_output_expr);
      break;
    } case '+': {
      differentiateExpr(grad_info, expr->tree(0), grad_output_expr);
      differentiateExpr(grad_info, expr->tree(1), grad_output_expr);
      break;
    } case '*': {
      differentiateExpr(grad_info, expr->tree(0),
                        Compound::create('*', expr->range(), {grad_output_expr, expr->tree(1)}));
      differentiateExpr(grad_info, expr->tree(1),
                        Compound::create('*', expr->range(), {grad_output_expr, expr->tree(0)}));
      break;
    } default:
      throw std::runtime_error("unsupported expr kind in AD: " + kindToString(expr->kind()));
  }
}

TreeRef gradNameOf(const Ident& name) {
  return Ident::create(dummyRange, std::string("d_") + name.name());
}

// TODO: check that there is no write after read
std::string differentiate(const std::string& source) {
  using namespace lang;
  auto def = Def(Sema().checkFunction(Parser(source).parseFunction()));

  GradInfo grad_info;
  // We will take grads w.r.t. returns as inputs, so mark them as present
  for (Param r : def.returns()) {
    grad_info.mark_has_grad(r.ident().name());
  }
  auto body = def.statements();
  auto it = body.end();
  if (it == body.begin())
    throw std::runtime_error("empty body");
  do {
    Comprehension comp = *(--it);
    // TODO: check assignment
    auto primal_output = comp.ident();
    if (grad_info.has_zero_grad(primal_output.name()))
      continue;
    auto grad_output_expr = Access::create(dummyRange,
                                           gradNameOf(primal_output),
                                           comp.indices());
    differentiateExpr(grad_info, comp.rhs(), grad_output_expr);
    if (comp.assignment()->kind() == TK_PLUS_EQ_B) {
      grad_info.mark_zero_grad(primal_output.name());
    }
  } while (it != body.begin());

  auto inferredType = Compound::create(TK_INFERRED, dummyRange, {});

  std::vector<TreeRef> reverseInputs;
  for (Param input : def.params()) {
    reverseInputs.push_back(input);
  }
  for (Param output : def.returns()) {
    reverseInputs.push_back(output);
  }
  for (Param output : def.returns()) {
    reverseInputs.push_back(Param::create(dummyRange, gradNameOf(output.ident()), inferredType));
  }

  std::vector<TreeRef> reverseOutputs;
  for (Param input : def.params()) {
    reverseOutputs.push_back(Param::create(dummyRange, gradNameOf(input.ident()), inferredType));
  }

  auto reverseDef = Def::create(dummyRange,
                                Ident::create(dummyRange, "grad_" + def.name().name()),
                                ListView<Param>::create(dummyRange, std::move(reverseInputs)),
                                ListView<Param>::create(dummyRange, std::move(reverseOutputs)),
                                ListView<Comprehension>::create(dummyRange, grad_info.getGradComps()));
  std::ostringstream s;
  tcFormat(s, reverseDef);
  return s.str();
}

PYBIND11_MODULE(tc, m) {
  m.def("_differentiate", differentiate);
  m.def("set_logtostderr", [](bool logtostderr) {
    FLAGS_logtostderr = logtostderr;
  });
  m.def("set_debug_lang", [](bool debug_lang) {
    tc::FLAGS_debug_lang = debug_lang;
  });
  m.def("set_debug_halide", [](bool debug_halide) {
    tc::FLAGS_debug_halide = debug_halide;
  });
  m.def("set_debug_tc_mapper", [](bool debug_tc_mapper) {
    tc::FLAGS_debug_tc_mapper = debug_tc_mapper;
  });
  m.def("set_debug_cuda", [](bool debug_cuda) {
    tc::FLAGS_debug_cuda = debug_cuda;
  });
  m.def("set_debug_tuner", [](bool debug_tuner) {
    tc::FLAGS_debug_tuner = debug_tuner;
  });
  m.def(
      "set_dump_cuda", [](bool dump_cuda) { tc::FLAGS_dump_cuda = dump_cuda; });

  py::object dlpack;
  try {
    dlpack = py::module::import("torch.utils.dlpack");
  } catch (std::exception& e) {
    std::cerr << "\n PyTorch installation is missing, binary will be useless \n"
              << e.what() << std::endl;
  }
  py::class_<ATenCudaCompilationUnit>(m, "ATenCompilationUnit")
      .def(py::init<>())
      .def("define", &ATenCudaCompilationUnit::define, "Define the TC language")
      .def(
          "compile",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              py::list& inputs,
              const tc::MappingOptions& options) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            return instance.compile(name, atInputs, options);
          })
      .def(
          "run",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              py::list& inputs,
              py::list& outputs,
              size_t handle) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            std::vector<at::Tensor> atOutputs = getATenTensors(outputs, dlpack);
            instance.run(name, atInputs, atOutputs, handle);
            if (py::len(outputs) == 0) {
              convertToPyObjects(atOutputs, dlpack, outputs);
            }
          })
      .def(
          "uncheckedRun",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              py::list& inputs,
              py::list& outputs,
              size_t handle) {
            CHECK_LT(0, outputs.size());
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            std::vector<at::Tensor> atOutputs = getATenTensors(outputs, dlpack);
            instance.uncheckedRun(atInputs, atOutputs, handle);
          })
      .def(
          "inject_cuda",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              const std::string& injectedKernelName,
              const std::string& cudaSource,
              py::list& inputs,
              std::vector<uint64_t> grid,
              std::vector<uint64_t> block) {
            tc::ManualCudaCache::enableCache();
            tc::MappingOptions options =
                tc::MappingOptions::makeNaiveMappingOptions();
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            auto tensorsPair = tc::toConstDlpackTensors(atInputs);
            tc::ScopeGuard g(
                [&]() { tc::deleteDlmTensors(tensorsPair.second); });
            auto outTensorInfo = instance.inferOutputTensorInfo(name, atInputs);
            tc::ManualCudaCache::getCache()->cacheKernel(
                name,
                tensorsPair.first,
                outTensorInfo,
                injectedKernelName,
                {},
                cudaSource,
                tc::Grid(grid),
                tc::Block(block));
          });
}

} // namespace python
} // namespace tc

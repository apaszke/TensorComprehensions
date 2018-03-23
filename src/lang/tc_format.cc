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
#include "tc/lang/tc_format.h"

namespace lang {

namespace {

void showExpr(std::ostream& s, const TreeRef& expr);

std::ostream& operator<<(std::ostream& s, const Ident& id) {
  s << id.name();
}

std::ostream& operator<<(std::ostream& s, const Param& p) {
  if (!p.typeIsInferred()) {
    TensorType type {p.type()};
    s << kindToString(type.scalarType()) << "(";
    for (const TreeRef & dim_expr : p.dims())
      showExpr(s, dim_expr);
    s << ") ";
  }
  return s << p.ident();
}

template<typename T>
std::ostream& operator<<(std::ostream& s, const ListView<T>& list) {
  bool first = true;
  for (auto & elem : list) {
    if (!first)
      s << ", ";
    s << elem;
    first = false;
  }
  return s;
}

void showExpr(std::ostream& s, const TreeRef& expr) {
  switch (expr->kind()) {
    case TK_IDENT:
      s << Ident(expr);
    default:
      throw std::runtime_error("Unexpected kind in showExpr: " + kindToString(expr->kind()));
  }
}

} // anonymous namespace

std::string tcFormat(TreeRef def) {
  std::ostringstream ss;
  ss << "def " << def.name() << "(" << def.params() << ")" <<
     << " -> (" << def.returns() << ") {\n";
}

} // namespace lang


/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/callback.h"

#include "common/assert.h"
#include "dynarmic/backend/x64/block_of_code.h"

namespace Dynarmic::Backend::X64 {

Callback::~Callback() = default;

void SimpleCallback::EmitCall(BlockOfCode& code, std::function<void(RegList)> l) const {
    l({code.ABI_PARAM1, code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    code.CallFunction(fn);
}

void SimpleCallback::EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> l) const {
    l(code.ABI_PARAM1, {code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    code.CallFunction(fn);
}

ArgCallback ArgCallback::ArgFromJitState(u32 offset, u64 jit_state_value) const {
    ASSERT(jit_state_value == arg);
    ArgCallback result = *this;
    result.arg_offset = offset;
    return result;
}

void ArgCallback::EmitLoadArg(BlockOfCode& code, Xbyak::Reg64 reg) const {
    if (arg_offset) {
        code.mov(reg, code.qword[code.ABI_JIT_PTR + *arg_offset]);
    } else {
        code.mov(reg, arg);
    }
}

void ArgCallback::EmitCall(BlockOfCode& code, std::function<void(RegList)> l) const {
    l({code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    EmitLoadArg(code, code.ABI_PARAM1);
    code.CallFunction(fn);
}

void ArgCallback::EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> l) const {
#if defined(WIN32) && !defined(__MINGW64__)
    l(code.ABI_PARAM2, {code.ABI_PARAM3, code.ABI_PARAM4});
    EmitLoadArg(code, code.ABI_PARAM1);
#else
    l(code.ABI_PARAM1, {code.ABI_PARAM3, code.ABI_PARAM4});
    EmitLoadArg(code, code.ABI_PARAM2);
#endif
    code.CallFunction(fn);
}

}  // namespace Dynarmic::Backend::X64

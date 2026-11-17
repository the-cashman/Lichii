// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/debugger/debugger.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc.h"

namespace Kernel {

namespace {

/// EDEN_NX2_CORE_LOCK: the core context lock is a KSpinLock and Idle sleeps on an address wait.
/// The std::mutex and condition variable went through winpthreads: every contended lock, every
/// Interrupt of an idle core and every RunThread entry and exit paid semaphore round trips.
bool CoreLockEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CORE_LOCK");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

class ScopedCoreGuard {
public:
    explicit ScopedCoreGuard(KSpinLock& fast_guard, std::mutex& guard)
        : m_fast_guard{fast_guard}, m_guard{guard}, m_fast{CoreLockEnabled()} {
        if (m_fast) {
            m_fast_guard.Lock();
        } else {
            m_guard.lock();
        }
    }

    ~ScopedCoreGuard() {
        if (m_fast) {
            m_fast_guard.Unlock();
        } else {
            m_guard.unlock();
        }
    }

    LICHII_NON_COPYABLE(ScopedCoreGuard);
    LICHII_NON_MOVEABLE(ScopedCoreGuard);

private:
    KSpinLock& m_fast_guard;
    std::mutex& m_guard;
    const bool m_fast;
};
} // Anonymous namespace

PhysicalCore::PhysicalCore(KernelCore& kernel, std::size_t core_index)
    : m_core_index{core_index}, m_idle_stats{std::getenv("EDEN_NX2_IDLE_STATS") != nullptr}
{
    m_is_single_core = !kernel.IsMulticore();
}
PhysicalCore::~PhysicalCore() = default;

void PhysicalCore::RunThread(KernelCore& kernel, Kernel::KThread* thread) {
    auto* process = thread->GetOwnerProcess();
    auto& system = kernel.System();
    auto* interface = process->GetArmInterface(m_core_index);

    interface->Initialize();

    const auto EnterContext = [&]() {
        // Lock the core context.
        ScopedCoreGuard lk{m_fast_guard, m_guard};

        // Check if we are already interrupted. If we are, we can just stop immediately.
        if (m_is_interrupted) {
            return false;
        }

        // Mark that we are running.
        m_arm_interface = interface;
        m_current_thread = thread;

        // Acquire the lock on the thread parameters.
        // This allows us to force synchronization with Interrupt.
        interface->LockThread(thread);

        return true;
    };

    const auto ExitContext = [&]() {
        // Save the JIT context back to the thread so the debugger can
        // read the current register state. Debug halt paths (step,
        // breakpoint, watchpoint) return from RunThread without going
        // through the scheduler's Unload/SaveContext.
        if (system.DebuggerEnabled()) {
            interface->GetContext(thread->GetContext());
        }

        // Unlock the thread.
        interface->UnlockThread(thread);

        // Lock the core context.
        ScopedCoreGuard lk{m_fast_guard, m_guard};

        // On exit, we no longer are running.
        m_arm_interface = nullptr;
        m_current_thread = nullptr;
    };

    while (true) {
        // If the thread is scheduled for termination, exit.
        if (thread->HasDpc() && thread->IsTerminationRequested()) {
            thread->Exit(kernel);
        }

        // Notify the debugger and go to sleep if a step was performed
        // and this thread has been scheduled again.
        if (thread->GetStepState() == StepState::StepPerformed) {
            system.GetDebugger().NotifyThreadStopped(thread);
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Otherwise, run the thread.
        Core::HaltReason hr{};
        {
            // If we were interrupted, exit immediately.
            if (!EnterContext()) {
                return;
            }

            if (thread->GetStepState() == StepState::StepPending) {
                hr = interface->StepThread(thread);

                if (True(hr & Core::HaltReason::StepThread)) {
                    thread->SetStepState(StepState::StepPerformed);
                }
            } else {
                hr = interface->RunThread(thread);
            }

            ExitContext();
        }

        // Determine why we stopped.
        // If a step completed successfully, skip other halt reason handlers
        // the step takes priority (e.g. step may also set InstructionBreakpoint
        // if the next instruction happens to be a breakpoint).
        const bool step_completed = True(hr & Core::HaltReason::StepThread)
                                    && thread->GetStepState() == StepState::StepPerformed;
        const bool supervisor_call = !step_completed && True(hr & Core::HaltReason::SupervisorCall);
        const bool prefetch_abort = !step_completed && True(hr & Core::HaltReason::PrefetchAbort);
        const bool breakpoint = !step_completed && True(hr & Core::HaltReason::InstructionBreakpoint);
        const bool data_abort = !step_completed && True(hr & Core::HaltReason::DataAbort);
        const bool interrupt = !step_completed && True(hr & Core::HaltReason::BreakLoop);

        // Since scheduling may occur here, we cannot use any cached
        // state after returning from calls we make.

        // Notify the debugger and go to sleep if a breakpoint was hit,
        // or if the thread is unable to continue for any reason.
        if (breakpoint || prefetch_abort) {
            if (breakpoint) {
                interface->RewindBreakpointInstruction();
                // RewindBreakpointInstruction sets the JIT state to the
                // saved breakpoint context. Update the thread context to
                // match, since ExitContext already saved the post-execution
                // state.
                interface->GetContext(thread->GetContext());
            }
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadStopped(thread);
            } else {
                interface->LogBacktrace(process);
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Notify the debugger and go to sleep on data abort.
        if (data_abort) {
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadWatchpoint(thread, *interface->HaltedWatchpoint());
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Handle system calls.
        if (supervisor_call) {
            // Perform call.
            Svc::Call(system, interface->GetSvcNumber());
            return;
        }

        // Handle external interrupt sources.
        if (interrupt || m_is_single_core) {
            return;
        }
    }
}

void PhysicalCore::LoadContext(const KThread* thread) {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->SetContext(thread->GetContext());
        interface->SetTpidrroEl0(GetInteger(thread->GetTlsAddress()));
        interface->SetWatchpointArray(&process->GetWatchpoints());
    }
}

void PhysicalCore::LoadSvcArguments(const KProcess& process, std::span<const uint64_t, 8> args) {
    process.GetArmInterface(m_core_index)->SetSvcArguments(args);
}

void PhysicalCore::SaveContext(KThread* thread) const {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->GetContext(thread->GetContext());
    }
}

void PhysicalCore::SaveSvcArguments(KProcess& process, std::span<uint64_t, 8> args) const {
    process.GetArmInterface(m_core_index)->GetSvcArguments(args);
}

void PhysicalCore::CloneFpuStatus(KThread* dst) const {
    auto* process = dst->GetOwnerProcess();

    Svc::ThreadContext ctx{};
    process->GetArmInterface(m_core_index)->GetContext(ctx);

    dst->GetContext().fpcr = ctx.fpcr;
    dst->GetContext().fpsr = ctx.fpsr;
}

void PhysicalCore::LogBacktrace(KernelCore& kernel) {
    auto* process = GetCurrentProcessPointer(kernel);
    if (!process) {
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->LogBacktrace(process);
    }
}

void PhysicalCore::Idle(KernelCore& kernel) {
    // Guest job systems hand work between cores many times per frame; blocking at once adds the
    // host's thread wake-up latency to every handoff. EDEN_NX2_IDLE_SPIN_US=<n>: poll for up to
    // that many microseconds before blocking. Switch 2 titles poll 100 us by default outside Android
    // (title 1's frames: 42 ms without, 35 ms with 50 us, 34 ms with 100 us, 36 ms with 300 us).
    static const long long spin_setting = [] {
        const char* text = std::getenv("EDEN_NX2_IDLE_SPIN_US");
        return text ? std::strtoll(text, nullptr, 10) : -1LL;
    }();
    long long spin_us = spin_setting;
    if (spin_us < 0) {
#ifdef __ANDROID__
        spin_us = 0;
#else
        const KProcess* process = kernel.ApplicationProcess();
        spin_us = process != nullptr && process->IsSwitch2() ? 100 : 0;
#endif
    }
    // JIT warm-up: compile listed code on this core's thread, in slices that end as soon as a
    // thread is ready to run here (checked before every block), in place of the spin while
    // there is any.
    if (!m_is_single_core) {
        if (KProcess* const process = kernel.ApplicationProcess(); process != nullptr) {
            if (Core::ArmInterface* const arm = process->GetArmInterface(m_core_index)) {
                while (!m_is_interrupted.load(std::memory_order_acquire) &&
                       arm->WarmUp(m_is_interrupted)) {
                }
            }
        }
    }
    // That poll is a hardware wait on the interrupt flag (Common::WaitForFlag) rather than a spin
    // of yields, and only as long as recent idle periods suggest it pays (Common::NextPollWindow):
    // a core that keeps idling past the limit polls less, and blocks sooner.
    const std::chrono::nanoseconds limit = std::chrono::microseconds{spin_us};
    const auto start = std::chrono::steady_clock::now();
    bool polled = false;
    std::chrono::steady_clock::rep interrupt_time{};
    if (limit.count() > 0) {
        if (m_idle_poll.count() <= 0 || m_idle_poll > limit) {
            m_idle_poll = limit;
        }
        polled = Common::WaitForFlag(m_is_interrupted, start + m_idle_poll);
    }
    if (!polled) {
        if (CoreLockEnabled()) {
            while (m_is_interrupted.load(std::memory_order_acquire) == 0) {
                Nx2Wait::Wait(m_is_interrupted, 0);
            }
            if (m_idle_stats) {
                // Under the guard, as Interrupt writes it: no later interrupt can replace it yet.
                ScopedCoreGuard lk{m_fast_guard, m_guard};
                interrupt_time = m_interrupt_time.load(std::memory_order_relaxed);
            }
        } else {
            std::unique_lock lk{m_guard};
            m_on_interrupt.wait(lk, [this] { return m_is_interrupted.load() != 0; });
            // Under m_guard, as Interrupt writes it: no later interrupt can replace it yet.
            interrupt_time = m_interrupt_time.load(std::memory_order_relaxed);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const auto poll = m_idle_poll;
    if (limit.count() > 0) {
        m_idle_poll = Common::NextPollWindow(m_idle_poll, end - start, limit);
    }

    if (!m_idle_stats) {
        return;
    }
    if (m_stats_start == std::chrono::steady_clock::time_point{}) {
        m_stats_start = end;
    }
    ++m_stats_idles;
    m_stats_poll_sum += std::min<std::chrono::nanoseconds>(poll, end - start);
    if (polled) {
        ++m_stats_polled;
    } else {
        const std::chrono::steady_clock::time_point interrupt{
            std::chrono::steady_clock::duration{interrupt_time}};
        m_stats_blocked_wakes.push_back(end - std::max(interrupt, start));
    }
    if (end - m_stats_start < std::chrono::seconds{10}) {
        return;
    }
    auto& wakes = m_stats_blocked_wakes;
    const auto percentile = [&wakes](size_t percent) {
        if (wakes.empty()) {
            return 0.0;
        }
        const auto it = wakes.begin() + (wakes.size() - 1) * percent / 100;
        std::nth_element(wakes.begin(), it, wakes.end());
        return std::chrono::duration<double, std::micro>{*it}.count();
    };
    const double p50 = percentile(50);
    const double p99 = percentile(99);
    LOG_INFO(Core_ARM,
             "Guest core {} idle over {:.1f} s: {} periods, {} woken while polling, {:.1f} us "
             "polled on average; {} blocked, woken {:.1f} us (p50) {:.1f} us (p99) after the "
             "interrupt; poll now {:.1f} us",
             m_core_index, std::chrono::duration<double>{end - m_stats_start}.count(),
             m_stats_idles, m_stats_polled,
             std::chrono::duration<double, std::micro>{m_stats_poll_sum}.count() /
                 static_cast<double>(m_stats_idles),
             wakes.size(), p50, p99,
             std::chrono::duration<double, std::micro>{m_idle_poll}.count());
    m_stats_start = end;
    m_stats_idles = 0;
    m_stats_polled = 0;
    m_stats_poll_sum = {};
    wakes.clear();
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted.load() != 0;
}

void PhysicalCore::Interrupt() {
    if (CoreLockEnabled()) {
        Core::ArmInterface* arm_interface{};
        KThread* thread{};
        {
            ScopedCoreGuard lk{m_fast_guard, m_guard};
            arm_interface = m_arm_interface;
            thread = m_current_thread;
            if (m_idle_stats) {
                m_interrupt_time.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                                       std::memory_order_relaxed);
            }
            m_is_interrupted.store(1, std::memory_order_seq_cst);
            if (arm_interface != nullptr) {
                arm_interface->SignalInterrupt(thread);
            }
        }
        // Wakes Idle if it sleeps; costs no syscall otherwise.
        Nx2Wait::WakeAll(m_is_interrupted);
        return;
    }

    // Lock core context.
    std::scoped_lock lk{m_guard};

    // Load members.
    auto* arm_interface = m_arm_interface;
    auto* thread = m_current_thread;

    if (m_idle_stats) {
        m_interrupt_time.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                               std::memory_order_relaxed);
    }

    // Add interrupt flag.
    m_is_interrupted = 1;

    // Interrupt ourselves.
    m_on_interrupt.notify_one();

    // If there is no thread running, we are done.
    if (arm_interface == nullptr) {
        return;
    }

    // Interrupt the CPU.
    arm_interface->SignalInterrupt(thread);
}

void PhysicalCore::ClearInterrupt() {
    ScopedCoreGuard lk{m_fast_guard, m_guard};
    m_is_interrupted = 0;
}

} // namespace Kernel

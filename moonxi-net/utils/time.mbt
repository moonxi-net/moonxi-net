// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Simple stopwatch for timing code sections using a millisecond clock.
///
/// # Example
/// ```mbt nocheck
/// let sw = Stopwatch::new()
/// sw.start(fn() { 0.0 }) // pass a clock function returning current time in ms
/// // ... do work ...
/// let elapsed = sw.elapsed_s(fn() { 0.0 })
/// ```
pub(all) struct Stopwatch {
  mut start_ms : Float
}

///|
pub fn Stopwatch::new() -> Stopwatch {
  { start_ms: 0.0 }
}

///|
/// Start or restart the stopwatch
pub fn Stopwatch::start(self : Stopwatch, get_time_ms : () -> Float) -> Unit {
  self.start_ms = get_time_ms()
}

///|
/// Elapsed time in milliseconds since start
pub fn Stopwatch::elapsed_ms(
  self : Stopwatch,
  get_time_ms : () -> Float,
) -> Float {
  get_time_ms() - self.start_ms
}

///|
/// Elapsed time in seconds since start
pub fn Stopwatch::elapsed_s(
  self : Stopwatch,
  get_time_ms : () -> Float,
) -> Float {
  self.elapsed_ms(get_time_ms) / 1000.0
}

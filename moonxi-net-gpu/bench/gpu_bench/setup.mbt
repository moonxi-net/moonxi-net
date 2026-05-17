// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn init {
  @cuda.gpu_init(0)
  @gt.gpu_enable_pool(6_442_450_944L)
}

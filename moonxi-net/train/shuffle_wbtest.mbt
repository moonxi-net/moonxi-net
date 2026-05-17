// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
test "fisher_yates_shuffle returns array of same length" {
  let arr = [10, 20, 30, 40, 50]
  let shuffled = fisher_yates_shuffle(arr)
  assert_eq(shuffled.length(), 5)
  // Original should be unchanged
  assert_eq(arr.length(), 5)
  assert_eq(arr[0], 10)
  assert_eq(arr[4], 50)
}

///|
test "fisher_yates_shuffle returns a permutation (all elements present)" {
  let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let shuffled = fisher_yates_shuffle(arr)
  assert_eq(shuffled.length(), 10)
  // Every element from original must be in shuffled
  for v in arr {
    assert_true(shuffled.contains(v))
  }
  // Every element from shuffled must be in original
  for v in shuffled {
    assert_true(arr.contains(v))
  }
}

///|
test "fisher_yates_shuffle handles single element" {
  let arr = [42]
  let shuffled = fisher_yates_shuffle(arr)
  assert_eq(shuffled.length(), 1)
  assert_eq(shuffled[0], 42)
}

///|
test "fisher_yates_shuffle handles two elements" {
  let arr = [1, 2]
  let shuffled = fisher_yates_shuffle(arr)
  assert_eq(shuffled.length(), 2)
  assert_true(shuffled.contains(1))
  assert_true(shuffled.contains(2))
}

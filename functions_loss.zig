const std = @import("std");

pub fn CategoricalCrossentropy(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u8) F {
      return -@log(predictions[target]);
    }

    pub fn backward(predictions: *const [LEN]F, target: u8, output: *[LEN]F) void {
      std.debug.assert(target < LEN);
      inline for (0..LEN) |i| {
        output[i] = if (i != target) 0 else -1/predictions[i];
      }
    }
  };
}

pub fn MeanSquaredError(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: *const [LEN]F) F {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        const diff = predictions[i] - target[i];
        sum += diff * diff;
      }
      return sum / LEN;
    }

    pub fn backward(predictions: *const [LEN]F, target: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = 2 * (predictions[i] - target[i]);
      }
    }
  };
}

pub fn MeanAbsoluteError(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: *const [LEN]F) F {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        const diff = predictions[i] - target[i];
        sum += @abs(diff);
      }
      return sum / LEN;
    }

    pub fn backward(predictions: *const [LEN]F, target: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (predictions[i] > target[i]) 1 else -1;
      }
    }
  };
}


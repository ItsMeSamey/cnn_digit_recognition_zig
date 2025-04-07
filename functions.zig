const std = @import("std");
const logger = @import("logger.zig");

// Activation Functions //

pub fn ReLU(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (input[i] < 0) 0 else input[i];
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (cache[i] == 0) 0 else derivative[i];
      }
    }
  };
}

pub fn Softmax(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      var sum_exp: F = 0;
      inline for (0..LEN) |i| {
        output[i] = @exp(input[i]);
        sum_exp += output[i];
      }
      inline for (0..LEN) |i| {
        output[i] /= sum_exp;
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = cache[i] * (1 - cache[i]) * derivative[i];
      }
    }
  };
}

pub fn Tanh(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = std.math.tanh(input[i]);
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = (1 - cache[i] * cache[i]) * derivative[i];
      }
    }
  };
}

pub fn Sigmoid(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = 1.0 / (1.0 + @exp(-input[i]));
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = cache[i] * (1 - cache[i]) * derivative[i];
      }
    }
  };
}

// Loss Functions //

pub fn CategoricalCrossentropy(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u32) F {
      std.debug.assert(target < LEN);
      logger.log(@src(), "predictions: {any}\n", .{predictions});
      // std.time.sleep(std.time.ns_per_ms * 500);
      return -@log(predictions[target]);
    }

    pub fn backward(predictions: *const [LEN]F, target: u32, output: *[LEN]F) void {
      std.debug.assert(target < LEN);
      inline for (0..LEN) |i| {
        output[i] = if (i != target) 0 else -1/predictions[i];
      }
    }
  };
}


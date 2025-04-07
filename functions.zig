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

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      _ = cache_in;
      inline for (0..LEN) |i| {
        output[i] = if (cache_out[i] == 0) 0 else derivative[i];
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

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      _ = cache_in;
      inline for (0..LEN) |i| {
        output[i] = (1 - cache_out[i] * cache_out[i]) * derivative[i];
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

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      _ = cache_in;
      inline for (0..LEN) |i| {
        output[i] = cache_out[i] * (1 - cache_out[i]) * derivative[i];
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

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      _ = cache_in;
      inline for (0..LEN) |i| {
        output[i] = cache_out[i] * (1 - cache_out[i]) * derivative[i];
      }
    }
  };
}

pub fn Normalize(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        sum += input[i];
      }
      inline for (0..LEN) |i| {
        output[i] = input[i] / sum;
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = (cache_out[i] / cache_in[i] - cache_out[i]) * derivative[i];
      }
    }
  };
}

// Loss Functions //

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


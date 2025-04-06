const std = @import("std");
const math = std.math;

// Activation Functions //

pub fn ReLU(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (input[i] < 0) 0 else input[i];
      }
    }

    pub fn backward(derivative: *[LEN]F, cache: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (cache == 0) 0 else derivative;
      }
    }
  };
}

pub fn Softmax(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *[LEN]F, output: *[LEN]F) void {
      var sum_exp: F = 0;
      inline for (0..LEN) |i| {
        output[i] = math.exp(input[i]);
        sum_exp += output[i];
      }
      inline for (0..LEN) |i| {
        output[i] /= sum_exp;
      }
    }

    pub fn backward(derivative: *[LEN]F, cache: *[LEN]F, output: *[LEN]F) void {
      var sum_dot_derivative: F = 0;
      inline for (0..LEN) |i| {
        sum_dot_derivative += cache[i] * derivative[i];
      }
      inline for (0..LEN) |i| {
        output[i] = cache[i] * (derivative[i] - sum_dot_derivative);
      }
    }
  };
}

pub fn Tanh(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = math.tanh(input[i]);
      }
    }

    pub fn backward(derivative: *[LEN]F, cache: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = derivative[i] * (1 - cache[i] * cache[i]);
      }
    }
  };
}

pub fn Sigmoid(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = 1.0 / (1.0 + math.exp(-input[i]));
      }
    }

    pub fn backward(derivative: *[LEN]F, cache: *[LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = derivative[i] * cache[i] * (1 - cache[i]);
      }
    }
  };
}

// Loss Functions //

pub fn CategoricalCrossentropy(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *[LEN]F, target: u32) F {
      std.debug.assert(target < LEN);
      return -@log(predictions[target]);
    }

    pub fn backward(predictions: *[LEN]F, target: u32, output: *[LEN]F) void {
      std.debug.assert(target < LEN);
      inline for (0..LEN) |i| {
        output[i] = if (i != target) 0 else 1/predictions[i];
      }
    }
  };
}


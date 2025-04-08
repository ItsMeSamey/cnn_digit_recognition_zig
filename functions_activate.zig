const std = @import("std");

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

pub fn getPReLU(alpha: comptime_float) fn (LEN: comptime_int, F: type) type {
  std.debug.assert(alpha != 1);
  return struct {
    pub fn PReLU(LEN: comptime_int, F: type) type {
      return struct {
        pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
          inline for (0..LEN) |i| {
            output[i] = if (input[i] < 0) input[i] * @as(F, alpha) else input[i];
          }
        }

        pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
          _ = cache_out;
          inline for (0..LEN) |i| {
            output[i] = if (cache_in[i] < 0) @as(F, alpha) * derivative[i] else derivative[i];
          }
        }
      };
    }
  }.PReLU;
}

pub fn getELU(alpha: comptime_float) fn (LEN: comptime_int, F: type) type {
  return struct {
    pub fn ELU(LEN: comptime_int, F: type) type {
      return struct {
        pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
          inline for (0..LEN) |i| {
            output[i] = if (input[i] < 0) @as(F, alpha) * (std.math.exp(input[i]) - 1) else input[i];
          }
        }

        pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
          inline for (0..LEN) |i| {
            output[i] = if (cache_in[i] < 0) (cache_out[i] + @as(F, alpha)) * derivative[i] else derivative[i];
          }
        }
      };
    }
  }.ELU;
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

pub fn NormalizeAbsolute(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        sum += @abs(input[i]);
      }
      inline for (0..LEN) |i| {
        output[i] = input[i] / sum;
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = (cache_out[i] / cache_in[i] - @abs(cache_out[i])) * derivative[i];
      }
    }
  };
}

pub fn NormalizeSquared(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(input: *const [LEN]F, output: *[LEN]F) void {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        output[i] = input[i] * input[i];
        sum += output[i];
      }
      inline for (0..LEN) |i| {
        output[i] /= sum;
      }
    }

    pub fn backward(derivative: *const [LEN]F, cache_in: *const [LEN]F, cache_out: *const [LEN]F, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = 2 * (cache_out[i] / cache_in[i]) * (1 - cache_out[i]) * derivative[i];
      }
    }
  };
}


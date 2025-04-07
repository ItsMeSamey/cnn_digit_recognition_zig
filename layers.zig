const std = @import("std");
const logger = @import("logger.zig");

pub const LayerType = fn (F: type, in_training: bool, width: comptime_int, height: comptime_int) LayerOutputType;
pub const LayerOutputType = struct {
  width: comptime_int,
  height: comptime_int,
  // layer type must have the following functions
  // pub fn init() @This()
  // pub fn reset(self: *@This(), rng: std.Random) void
  // pub fn asBytes(self: *@This()) [<any_size>]u8
  // pub fn fromBytes([<any_size>]u8) @This()
  // pub fn forward(self: *@This(), input: *[height][width]F, output: *[out_height][out_width]F)
  // pub fn backward(
  //   self: *@This(),
  //   cache_in: *const [height][width]F,
  //   cache_out: *const [out_height][out_width]F,
  //   d_prev: *[height][width]F,
  //   d_next: *const [out_height][out_width]F,
  // ) Gradient
  // pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void
  layer:  type
};

pub fn getConvolver(filter_x: comptime_int, filter_y: comptime_int, stride_x: comptime_int, stride_y: comptime_int) LayerType {
  @setEvalBranchQuota(1000_000);
  std.debug.assert(filter_x >= 1);
  std.debug.assert(filter_y >= 1);
  std.debug.assert(stride_x >= 1);
  std.debug.assert(stride_y >= 1);

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      comptime var out_width = (width - filter_x) / stride_x + 1;
      if (out_width * stride_x < width) out_width += 1;
      comptime var out_height = (height - filter_y) / stride_y + 1;
      if (out_height * stride_y < height) out_height += 1;
      const as_byte_size = (filter_x * filter_y + 1) * @sizeOf(F);

      const Gradient = struct {
        filter: [filter_y][filter_x]F,
        bias: F,

        pub fn init() @This() {
          return .{
            .filter = undefined,
            .bias = undefined,
          };
        }

        pub fn reset(self: *@This()) void {
          self.bias = 0;
          inline for (0..filter_y) |i| {
            inline for (0..filter_x) |j| {
              self[i][j] = 0;
            }
          }
        }

        pub fn add(self: *@This(), other: *const @This()) void {
          self.bias += other.bias;
          inline for (0..filter_y) |i| {
            inline for (0..filter_x) |j| {
              self[i][j] += other[i][j];
            }
          }
        }

        // To calculate the average after adding n
        pub fn div(self: *@This(), n: F) void {
          self.bias /= n;
          inline for (0..filter_y) |i| {
            inline for (0..filter_x) |j| {
              self[i][j] /= n;
            }
          }
        }
      };

      const Layer = struct {
        filter: [filter_y][filter_x]F,
        bias: F,
        // input: if (in_training) *[width*height]F else void = if (in_training) undefined else {},
        // output: if (in_training) [width*height]F else void = if (in_training) undefined else {},
        pub fn init() @This() {
          return .{
            .filter = undefined,
            .bias = undefined,
          };
        }

        pub fn reset(self: *@This(), rng: std.Random) void {
          self.bias = 0;
          inline for (0..filter_x) |i| {
            inline for (0..filter_y) |j| {
              self.filter[i][j] = rng.float(F) - 0.5;
            }
          }
        }

        pub fn asBytes(self: *@This()) [as_byte_size]u8 {
          var bytes: [as_byte_size]u8 = undefined;
          @memcpy(bytes[0..filter_x * filter_y * @sizeOf(F)], std.mem.asBytes(&self.filter));
          @memcpy(bytes[filter_x * filter_y * @sizeOf(F)..], std.mem.asBytes(&self.bias));
          return bytes;
        }

        pub fn fromBytes(bytes: *[as_byte_size]u8) @This() {
          var self = @This().init();
          @memcpy(std.mem.asBytes(&self.filter), bytes[0..filter_x * filter_y * @sizeOf(F)]);
          @memcpy(std.mem.asBytes(&self.bias), bytes[filter_x * filter_y * @sizeOf(F)..]);
          return self;
        }

        pub fn forward(self: *@This(), input: *[height][width]F, output: *[out_height][out_width]F) void {
          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              var sum: F = self.bias;
              inline for (0..filter_y) |filter_y_offset| {
                inline for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (comptime in_y >= height and in_x >= width) continue;
                  sum += input[in_y][in_x] * self.filter[filter_y_offset][filter_x_offset];
                }
              }
              output[out_y][out_x] = sum;
            }
          }
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [height][width]F,
          cache_out: *const [out_height][out_width]F,
          d_prev: *[height][width]F,
          d_next: *const [out_height][out_width]F,
        ) Gradient {
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");
          _ = cache_out;
          var gradient = Gradient.init();
          gradient.reset();

          // Gradient with respect to the bias
          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              gradient.bias += d_next[out_y][out_x];
            }
          }

          inline for (0..height) |y| {
            inline for (0..width) |x| {
              d_prev[y][x] = 0;
            }
          }

          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              inline for (0..filter_y) |filter_y_offset| {
                inline for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (comptime in_y >= height and in_x >= width) continue;
                  // Gradient with respect to the filter
                  gradient.filter[filter_y_offset][filter_x_offset] += d_next[out_y][out_x] * cache_in[in_y][in_x];

                  // Gradient with respect to the input (d_prev)
                  d_prev[in_y][in_x] += d_next[out_y][out_x] * self.filter[filter_y_offset][filter_x_offset];
                }
              }
            }
          }

          return gradient;
        }

        pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void {
          self.bias += learning_rate * gradient.bias;
          inline for (0..out_width) |i| {
            inline for (0..width) |j| {
              self.filter[i][j] += learning_rate * gradient.filter[i][j];
            }
          }
        }
      };

      return .{
        .width = out_width,
        .height = out_height,
        .layer = Layer,
      };
    }
  }.getLayer;
}

test getConvolver {
  const LayerFn = getConvolver(2,2,2,2);
  const Layer = LayerFn(f32, false, 33, 32);
  try std.testing.expect(Layer.width == 16);
  try std.testing.expect(Layer.height == 17);
}

fn NoOpGradient(F: type) type {
  return struct {
    pub fn init() @This() {
      return .{};
    }

    pub fn reset(_: *@This()) void {
      // NoOp
    }

    pub fn add(_: *@This(), _: *const @This()) void {
      // NoOp
    }

    pub fn div(_: *@This(), _: F) void {
      // NoOp
    }
  };
}

pub fn getMaxPooling(pool_size_x: comptime_int, pool_size_y: comptime_int, stride_x: comptime_int, stride_y: comptime_int) LayerType {
  @setEvalBranchQuota(1000_000);
  std.debug.assert(pool_size_x >= 1);
  std.debug.assert(pool_size_y >= 1);
  std.debug.assert(stride_x >= 1);
  std.debug.assert(stride_y >= 1);

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      comptime var out_width = (width - pool_size_x) / stride_x + 1;
      if (out_width * stride_x < width) out_width += 1;
      comptime var out_height = (height - pool_size_y) / stride_y + 1;
      if (out_height * stride_y < height) out_height += 1;

      const Gradient = NoOpGradient(F); 

      const Layer = struct {
        const idxType = std.meta.Int(.unsigned, std.math.log2(pool_size_y*pool_size_x));
        max_idx: if (in_training) [out_height][out_width]idxType else void = if (in_training) undefined else {},

        pub fn init() @This() {
          return .{};
        }

        pub fn reset(_: *@This(), _: std.Random) void {
          // Nothing to reset
        }
        pub fn asBytes(_: *@This()) [0]u8 {
          return [0]u8{}; // Nothing to save
        }

        pub fn fromBytes(_: [0]u8) @This() {
          return .{}; // Nothing to restore
        }

        pub fn forward(self: *@This(), input: *[height][width]F, output: *[out_height][out_width]F) void {
          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              var max_val = -std.math.inf(F);
              var max_index: if (in_training) idxType else void = if (in_training) undefined else {};
              inline for (0..pool_size_y) |pool_y| {
                inline for (0..pool_size_x) |pool_x| {
                  const in_y = out_y * stride_y + pool_y;
                  const in_x = out_x * stride_x + pool_x;
                  if (comptime in_y >= height and in_x >= width) continue;

                  if (input[in_y][in_x] > max_val) {
                    max_val = input[in_y][in_x];
                    if (in_training) {
                      max_index = @as(idxType, pool_y * pool_size_x + pool_x);
                    }
                  }
                }
              }
              output[out_y][out_x] = max_val;

              if (comptime !in_training) continue;
              self.max_idx[out_y][out_x] = max_index;
            }
          }
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [height][width]F,
          cache_out: *const [out_height][out_width]F,
          d_prev: *[height][width]F,
          d_next: *const [out_height][out_width]F,
        ) Gradient {
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");
          _ = cache_in;
          _ = cache_out;

          inline for (0..height) |y| {
            inline for (0..width) |x| {
              d_prev[y][x] = 0;
            }
          }

          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              const max_index = self.max_idx[out_y][out_x];
              const prev_y = out_y * stride_y + @divTrunc(max_index, pool_size_x);
              const prev_x = out_x * stride_x + @rem(max_index, pool_size_x);
              if (prev_y < height and prev_x < width) {
                d_prev[prev_y][prev_x] += d_next[out_y][out_x];
              }
            }
          }

          return .{};
        }

        pub fn applyGradient(_: *@This(), _: *const Gradient, _: F) void {
          // NoOp
        }
      };

      return .{
        .width = out_width,
        .height = out_height,
        .layer = Layer,
      };
    }
  }.getLayer;
}

test getMaxPooling {
  const LayerFn = getMaxPooling(2, 2, 2, 2);
  const LayerTypeActual = LayerFn(f32, false, 4, 4);
  const Layer = LayerTypeActual.layer;
  var layer = Layer.init();

  var input: [4][4]f32 = .{
    .{ 1, 2, 3, 4 },
    .{ 5, 6, 7, 8 },
    .{ 9, 10, 11, 12 },
    .{ 13, 14, 15, 16 },
  };

  var output: [2][2]f32 = undefined;
  layer.forward(&input, &output);

  try std.testing.expectEqual(output[0][0], 6.0);
  try std.testing.expectEqual(output[0][1], 8.0);
  try std.testing.expectEqual(output[1][0], 14.0);
  try std.testing.expectEqual(output[1][1], 16.0);
}

// A special flatten layer (had to be implemented this way to prevent copying)
pub fn getFlattener() LayerType {
  @setEvalBranchQuota(1000_000);
  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      const out_width = width * height;

      const Layer = struct {
        pub fn forward(input: *[height][width]F) *[1][out_width]F {
          return @ptrCast(input);
        }

        pub fn backward(d_next: *[1][out_width]F) *[height][width]F {
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");
          return @ptrCast(d_next);
        }
      };

      return .{
        .width = out_width,
        .height = 1,
        .layer = Layer,
      };
    }
  }.getLayer;
}

test getFlattener {
  const LayerFn = getFlattener();
  const Layer = LayerFn(f32, false, 33, 32);
  try std.testing.expect(Layer.width == 33 * 32);
  try std.testing.expect(Layer.height == 1);
}

pub fn getDense(out_width: comptime_int, function_getter: fn(LEN: comptime_int, T: type) type) LayerType {
  @setEvalBranchQuota(1000_000);

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      std.debug.assert(height == 1);
      const Activate = function_getter(out_width, F);

      const as_byte_size = (width + 1) * out_width * @sizeOf(F);

      const Gradient = struct {
        weights: [out_width][width]F,
        biases: [out_width]F,

        pub fn init() @This() {
          return .{
            .weights = undefined,
            .biases = undefined,
          };
        }

        pub fn reset(self: *@This()) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            self.biases[i] = 0;
            inline for (0..width) |j| {
              self.weights[i][j] = 0;
            }
          }
        }

        pub fn add(self: *@This(), other: *const @This()) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            self.biases[i] += other.biases[i];
            inline for (0..width) |j| {
              self.weights[i][j] += other.weights[i][j];
            }
          }
        }

        pub fn div(self: *@This(), n: F) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            self.biases[i] /= n;
            inline for (0..width) |j| {
              self.weights[i][j] /= n;
            }
          }
        }
      };

      const Layer = struct {
        weights: [out_width][width]F,
        biases: [out_width]F,

        pub fn init() @This() {
          return .{
            .weights = undefined,
            .biases = undefined,
          };
        }

        pub fn reset(self: *@This(), rng: std.Random) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            self.biases[i] = rng.float(F) - 0.5;
            inline for (0..width) |j| {
              self.weights[i][j] = rng.float(F) - 0.5;
            }
          }
        }

        pub fn asBytes(self: *@This()) [as_byte_size]u8 {
          var bytes: [as_byte_size]u8 = undefined;
          @memcpy(bytes[0..width * out_width * @sizeOf(F)], std.mem.asBytes(&self.weights));
          @memcpy(bytes[width * out_width * @sizeOf(F)..], std.mem.asBytes(&self.biases));
          return bytes;
        }

        pub fn fromBytes(bytes: *[as_byte_size]u8) @This() {
          var self = @This().init();
          @memcpy(std.mem.asBytes(&self.weights), bytes[0..width * out_width * @sizeOf(F)]);
          @memcpy(std.mem.asBytes(&self.biases), bytes[width * out_width * @sizeOf(F)..]);
          return self;
        }

        pub fn forward(self: *@This(), input: *[1][width]F, output: *[1][out_width]F) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            var sum: F = self.biases[i];
            inline for (0..width) |j| {
              sum += input[0][j] * self.weights[i][j];
            }
            output[0][i] = sum;
            Activate.forward(&output[0], &output[0]);
          }
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [1][width]F,
          cache_out: *const [1][out_width]F,
          d_prev: *[1][width]F,
          d_next: *const [1][out_width]F,
        ) Gradient {
          @setEvalBranchQuota(1000_000);
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");
          var gradient = Gradient.init();

          // Gradient with respect to biases
          logger.log(&@src(), "{s}\n", .{@typeName(Activate)});
          // logger.log(&@src(), "d_next: {any}\n", .{d_next});
          // logger.log(&@src(), "cache: {any}\n", .{cache_out});
          Activate.backward(@ptrCast(d_next), @ptrCast(cache_out), &gradient.biases);
          // logger.log(&@src(), "D ({s})\n{any}\n", .{@typeName(@This()), gradient.biases});
          
          inline for (0..width) |j| {
            d_prev[0][j] = 0;
            inline for (0..out_width) |i| {
              gradient.weights[i][j] = gradient.biases[i] * cache_in[0][j]; // Gradient with respect to weights
              d_prev[0][j] += gradient.biases[i] * self.weights[i][j]; // Gradient with respect to the input (d_prev)
            }
          }

          // logger.log(&@src(), "d_prev: {any}\n", .{d_prev});
          return gradient;
        }

        pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..out_width) |i| {
            self.biases[i] -= learning_rate * gradient.biases[i];
            inline for (0..width) |j| {
              self.weights[i][j] += learning_rate * gradient.weights[i][j];
            }
          }
        }
      };

      return .{
        .width = out_width,
        .height = 1,
        .layer = Layer,
      };
    }
  }.getLayer;
}

test getDense {
  const LayerFn = getDense(5, @import("functions.zig").ReLU);
  const Layer = LayerFn(f32, true, 1, 10);
  try std.testing.expect(Layer.width == 5);
  try std.testing.expect(Layer.height == 1);
}


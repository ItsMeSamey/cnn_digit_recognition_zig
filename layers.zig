const std = @import("std");
const logger = @import("logger.zig");

pub const LayerType = fn (F: type, in_training: bool, width: comptime_int, height: comptime_int) LayerOutputType;
pub const LayerOutputType = struct {
  width: comptime_int,
  height: comptime_int,
  // layer type must have the following functions
  // pub fn reset(self: *@This(), rng: std.Random) void
  // pub fn forward(self: *@This(), input: *[height][width]F, output: *[out_height][out_width]F)
  // pub fn backward(
  //   self: *@This(),
  //   cache_in: *const [height][width]F,
  //   cache_out: *const [out_height][out_width]F,
  //   d_prev: *[height][width]F,
  //   d_next: *const [out_height][out_width]F,
  //   gradient: *Gradient,
  //   comptime calc_prev: bool,
  // ) void
  // pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void
  layer:  type
};


pub fn getConvolver(
  filter_x: comptime_int, filter_y: comptime_int,
  stride_x: comptime_int, stride_y: comptime_int,
  function_getter: fn(LEN: comptime_int, T: type) type
) LayerType {
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

      const Activate = function_getter(out_height * out_width, F);

      const Layer = struct {
        filter: [filter_y][filter_x]F,
        bias: F,
        cache_in: if (!in_training or @typeInfo(@TypeOf(Activate.backward)).@"fn".params[1].type.? == void) void else [out_width]F = undefined,

        const Gradient = struct {
          filter: [filter_y][filter_x]F,
          bias: F,

          pub fn reset(self: *@This()) void {
            self.bias = 0;
            for (0..filter_y) |i| {
              for (0..filter_x) |j| {
                self.filter[i][j] = 0;
              }
            }
          }

          pub fn add(self: *@This(), other: *const @This()) void {
            self.bias += other.bias;
            for (0..filter_y) |i| {
              for (0..filter_x) |j| {
                self.filter[i][j] += other.filter[i][j];
              }
            }
          }
        };

        pub fn reset(self: *@This(), rng: std.Random) void {
          self.bias = rng.float(F) - 0.5;
          for (0..filter_x) |i| {
            for (0..filter_y) |j| {
              self.filter[i][j] = rng.float(F)*10 - 5;
            }
          }
        }

        pub fn forward(self: *const @This(), input: *[height][width]F, output: *[out_height][out_width]F) void {
          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              var sum: F = self.bias;
              inline for (0..filter_y) |filter_y_offset| {
                inline for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (in_y < height and in_x < width) {
                    sum += input[in_y][in_x] * self.filter[filter_y_offset][filter_x_offset];
                  }
                }
              }

              if (@TypeOf(self.cache_in) == void) {
                output[out_y][out_x] = sum;
              } else {
                self.cache_in[out_y][out_x] = sum;
              }
            }
          }

          Activate.forward(@ptrCast(if (@TypeOf(self.cache_in) == void) output else &self.cache_in), @ptrCast(output));
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [height][width]F,
          cache_out: *const [out_height][out_width]F,
          d_prev: *[height][width]F,
          d_next: *const [out_height][out_width]F,
          gradient: *Gradient,
          comptime calc_prev: bool,
        ) void {
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");

          var dbuf: [out_height][out_width]F = undefined;
          Activate.backward(@ptrCast(d_next), if (@TypeOf(self.cache_in) == void) {} else @ptrCast(&self.cache_in), @ptrCast(cache_out), @ptrCast(&dbuf));
          // Gradient with respect to the bias
          inline for (0..out_height) |out_y| {
            inline for (0..out_width) |out_x| {
              gradient.bias += dbuf[out_y][out_x];
            }
          }

          if (calc_prev) {
            for (0..height) |y| {
              for (0..width) |x| {
                d_prev[y][x] = 0;
              }
            }
          }

          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              inline for (0..filter_y) |filter_y_offset| {
                inline for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (in_y < height and in_x < width) {
                    // Gradient with respect to the filter
                    gradient.filter[filter_y_offset][filter_x_offset] += d_next[out_y][out_x] * cache_in[in_y][in_x];

                    // Gradient with respect to the input (d_prev)
                    if (calc_prev) {
                      d_prev[in_y][in_x] += d_next[out_y][out_x] * self.filter[filter_y_offset][filter_x_offset];
                    }
                  }
                }
              }
            }
          }
        }

        pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void {
          self.bias -= learning_rate * gradient.bias;
          inline for (0..filter_y) |i| {
            inline for (0..filter_x) |j| {
              self.filter[i][j] -= learning_rate * gradient.filter[i][j];
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
  const LayerFn = getConvolver(2,2,2,2, @import("functions_activate.zig").getPReLU(0.1));
  const Layer = LayerFn(f32, false, 33, 32);
  try std.testing.expect(Layer.width == 16);
  try std.testing.expect(Layer.height == 17);
}


pub fn getMaxPooling(pool_size_x: comptime_int, pool_size_y: comptime_int, stride_x: comptime_int, stride_y: comptime_int) LayerType {
  @setEvalBranchQuota(1000_000);
  std.debug.assert(pool_size_x >= 1);
  std.debug.assert(pool_size_y >= 1);
  std.debug.assert(stride_x >= 1);
  std.debug.assert(stride_y >= 1);

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      const out_width = (width - pool_size_x) / stride_x + 1;
      const out_height = (height - pool_size_y) / stride_y + 1;

      const Layer = struct {
        pub const Gradient = void;

        const idxType = std.meta.Int(.unsigned, std.math.log2(pool_size_y*pool_size_x));
        max_idx: if (in_training) [out_height][out_width]idxType else void = if (in_training) undefined else {},

        pub fn reset(_: *@This(), _: std.Random) void {
          // Nothing to reset
        }

        pub fn forward(self: *const @This(), input: *[height][width]F, output: *[out_height][out_width]F) void {
          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              var max_val = -std.math.inf(F);
              var max_index: if (in_training) idxType else void = if (in_training) undefined else {};
              for (0..pool_size_y) |pool_y| {
                for (0..pool_size_x) |pool_x| {
                  const in_y = out_y * stride_y + pool_y;
                  const in_x = out_x * stride_x + pool_x;
                  if (in_y < height and in_x < width) {
                    if (input[in_y][in_x] > max_val) {
                      max_val = input[in_y][in_x];
                      if (in_training) {
                        max_index = @as(idxType, pool_y * pool_size_x + pool_x);
                      }
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
          gradient: *Gradient,
          comptime calc_prev: bool,
        ) void {
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");
          _ = cache_in;
          _ = cache_out;
          _ = gradient;

          if (!calc_prev) return;

          for (0..height) |y| {
            for (0..width) |x| {
              d_prev[y][x] = 0;
            }
          }

          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
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
  var layer: Layer = undefined;

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

// A special reshape layer (had to be implemented this way to prevent copying)
pub fn getReshaper(out_height: comptime_int, out_width: comptime_int) LayerType {
  @setEvalBranchQuota(1000_000);
  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      _ = in_training;
      if (comptime height*width != out_height*out_width) {
        @compileError(std.fmt.comptimePrint("Cant reshape to {d}x{d} from {d}x{d}", .{ out_height, out_width, height, width }));
      }

      return .{
        .width = out_width,
        .height = out_height,
        .layer = struct {
          pub fn forward(input: *[height][width]F) *[out_height][out_width]F {
            return @ptrCast(input);
          }
        },
      };
    }
  }.getLayer;
}


// A special flatten layer
pub fn getFlattener() LayerType {
  @setEvalBranchQuota(1000_000);
  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      return getReshaper(1, width * height)(F, in_training, height, width);
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

      const Layer = struct {
        weights: [out_width][width]F,
        biases: [out_width]F,
        cache_in: if (!in_training or @typeInfo(@TypeOf(Activate.backward)).@"fn".params[1].type.? == void) void else [out_width]F = undefined,

        pub const Gradient = struct {
          weights: [out_width][width]F,
          biases: [out_width]F,

          pub fn reset(self: *@This()) void {
            @setEvalBranchQuota(1000_000);
            for (0..out_width) |i| {
              self.biases[i] = 0;
              for (0..width) |j| {
                self.weights[i][j] = 0;
              }
            }
          }

          pub fn add(self: *@This(), other: *const @This()) void {
            @setEvalBranchQuota(1000_000);
            for (0..out_width) |i| {
              self.biases[i] += other.biases[i];
              for (0..width) |j| {
                self.weights[i][j] += other.weights[i][j];
              }
            }
          }
        };

        pub fn reset(self: *@This(), rng: std.Random) void {
          @setEvalBranchQuota(1000_000);
          for (0..out_width) |i| {
            self.biases[i] = rng.float(F) - 0.5;
            for (0..width) |j| {
              self.weights[i][j] = rng.float(F) - 0.5;
            }
          }
        }

        pub fn forward(self: if (in_training) *@This() else *const @This(), input: *[1][width]F, output: *[1][out_width]F) void {
          @setEvalBranchQuota(1000_000);
          logger.log(&@src(), "inp: {d}\n", .{input});
          defer logger.log(&@src(), "out: {d}\n", .{output});
          for (0..out_width) |i| {
            var sum: F = self.biases[i];
            for (0..width) |j| {
              sum += input[0][j] * self.weights[i][j];
            }
            if (@TypeOf(self.cache_in) == void) {
              output[0][i] = sum;
            } else {
              self.cache_in[i] = sum;
            }
          }

          Activate.forward(if (@TypeOf(self.cache_in) == void) &output[0] else &self.cache_in, &output[0]);
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [1][width]F,
          cache_out: *const [1][out_width]F,
          d_prev: *[1][width]F,
          d_next: *const [1][out_width]F,
          gradient: *Gradient,
          comptime calc_prev: bool,
        ) void {
          @setEvalBranchQuota(1000_000);
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");

          // Gradient with respect to biases
          logger.log(&@src(), "{s}\n", .{@typeName(Activate)});
          logger.log(&@src(), "d_next: {any}\n", .{d_next});
          logger.log(&@src(), "cache: {any}\n", .{cache_out});
          var biases: [out_width]F = undefined;
          Activate.backward(&d_next[0], if (@TypeOf(self.cache_in) == void) {} else &self.cache_in, &cache_out[0], &biases);
          // logger.log(&@src(), "D ({s})\n{any}\n", .{@typeName(@This()), gradient.biases});

          // Gradient with respect to weights
          for (0..out_width) |i| {
            gradient.biases[i] += biases[i];
            for (0..width) |j| {
              gradient.weights[i][j] += biases[i] * cache_in[0][j];
            }
          }

          if (!calc_prev) return;

          // Gradient with respect to the input (d_prev)
          for (0..width) |j| {
            d_prev[0][j] = 0;
            for (0..out_width) |i| {
              d_prev[0][j] += biases[i] * self.weights[i][j];
            }
          }

          logger.log(&@src(), "d_prev: {any}\n", .{d_prev});
        }

        pub fn applyGradient(self: *@This(), gradient: *const Gradient, learning_rate: F) void {
          @setEvalBranchQuota(1000_000);
          for (0..out_width) |i| {
            self.biases[i] -= learning_rate * gradient.biases[i];
            for (0..width) |j| {
              self.weights[i][j] -= learning_rate * gradient.weights[i][j];
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
  const LayerFn = getDense(5, @import("functions_activate.zig").ReLU);
  const Layer = LayerFn(f32, true, 1, 10);
  try std.testing.expect(Layer.width == 5);
  try std.testing.expect(Layer.height == 1);
}

fn GetLOPS(
  layers: anytype,
  F: type,
  in_training: bool,
  height: comptime_int,
  width: comptime_int,
  mode: enum {array, tuple},
) type {
  const LayerProperties = struct {
    output_height: comptime_int,
    output_width: comptime_int,
    layer_type: type,

    need_gradient: bool,
    gradient_type: type,

    const Self = @This();
    const SimpleWrapped = struct {
      self: Self,
      is_simple: bool,
    };

    fn fromLayerType(layer_output: LayerType, h_in: comptime_int, w_in: comptime_int) SimpleWrapped {
      const result = layer_output(F, in_training, h_in, w_in);
      const is_simple = @TypeOf(result.layer.forward) == fn (input: *[h_in][w_in]F) *[result.height][result.width]F;
      const gradient_type = if (is_simple) void else result.layer.Gradient;

      return .{
        .self = .{
          .output_height = result.height,
          .output_width = result.width,
          .layer_type = result.layer,
          .need_gradient = @sizeOf(gradient_type) != 0,
          .gradient_type = gradient_type,
        },
        .is_simple = is_simple,
      };
    }

    fn translateAll() []const @This() {
      comptime var translated_layers: []const SimpleWrapped = &.{};
      if (mode == .array) {
        inline for (layers) |layer| {
          translated_layers = translated_layers ++ &[_]SimpleWrapped{fromLayerType(
            layer,
            if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].self.output_height else height,
            if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].self.output_width else width,
          )};
        }
      } else if (mode == .tuple) {
        inline for (layers) |layer| {
          translated_layers = translated_layers ++ &[_]SimpleWrapped{fromLayerType(layer, height, width)};
        }
      }
      comptime var retval: []const @This() = &.{};
      inline for (translated_layers) |layer| {
        if (layer.is_simple) continue;
        retval = retval ++ &[_]@This(){layer.self};
      }
      return retval;
    }
  };

  const LayerOperationsType = struct {
    fn getType(L: []const LayerProperties) type {
      comptime var fields: []const std.builtin.Type.StructField = &.{};
      inline for (L, 0..) |l, i| {
        fields = fields ++ &[1]std.builtin.Type.StructField{.{
          .name = std.fmt.comptimePrint("{d}", .{i}),
          .type = l.layer_type,
          .default_value_ptr = null,
          .is_comptime = false,
          .alignment = @alignOf(l.layer_type),
        }};
      }
      return @Type(.{
        .@"struct" = .{
          .layout = .auto,
          .fields = fields,
          .decls = &[_]std.builtin.Type.Declaration{},
          .is_tuple = false,
        }
      });
    }

    const Layers = LayerProperties.translateAll();
    const LayerInstanceType = getType(Layers);

    fn getLayerInputDims(layer_num: comptime_int) [2]comptime_int {
      if (layer_num == 0) {
        const l = Layers[0];
        const inputPtrType = @typeInfo(@TypeOf(l.layer_type.forward)).@"fn".params[1].type.?;
        const inputType = std.meta.Child(inputPtrType);
        const h = @typeInfo(inputType).array.len;
        const subinputType = std.meta.Child(inputType);
        const w = @typeInfo(subinputType).array.len;
        return .{h, w};
      }
      const layer = Layers[layer_num - 1];
      return .{layer.output_height, layer.output_width};
    }

    fn getInputOffsets() [Layers.len]comptime_int {
      comptime var retval: [Layers.len]comptime_int = undefined;
      retval[0] = 0;
      inline for (1..Layers.len) |i| {
        retval[i] = retval[i-1] + Layers[i-1].output_height * Layers[i-1].output_width;
      }
      return retval;
    }
  };

  const GradientType = struct {
    sub: SubType,

    const SubType = init: {
      var fields: []const std.builtin.Type.StructField = &.{};
      for (LayerOperationsType.Layers, 0..) |l, i| {
        const gradient_type = if (l.need_gradient) l.gradient_type else void;
        fields = fields ++ &[1]std.builtin.Type.StructField{.{
          .name = std.fmt.comptimePrint("{d}", .{i}),
          .type = gradient_type,
          .default_value_ptr = null,
          .is_comptime = false,
          .alignment = @alignOf(gradient_type),
        }};
      }

      break :init @Type(.{
        .@"struct" = .{
          .layout = .auto,
          .fields = fields,
          .decls = &[_]std.builtin.Type.Declaration{},
          .is_tuple = false,
        }
      });
    };

    pub fn reset(self: *@This()) void {
      inline for (LayerOperationsType.Layers, 0..) |l, i| {
        if (l.need_gradient) @field(self.sub, std.fmt.comptimePrint("{d}", .{i})).reset();
      }
    }

    pub fn add(self: *@This(), other: *@This()) void {
      inline for (LayerOperationsType.Layers, 0..) |l, i| {
        const name = std.fmt.comptimePrint("{d}", .{i});
        if (l.need_gradient) @field(self.sub, name).add(&@field(other.sub, name));
      }
    }

    pub fn format(value: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
      _ = fmt;
      _ = options;
      inline for (LayerOperationsType.Layers, 0..) |l, i| {
        try std.fmt.format(writer, "\n-------- {s} --------\n{any}\n", .{@typeName(l.layer_type), @field(value.sub, std.fmt.comptimePrint("{d}", .{i}))});
      }
    }
  };

  return struct {
    const LayerOperations = LayerOperationsType;
    const Gradient = GradientType;
  };
}


pub fn mergeArray(layers: anytype) LayerType {
  if (@typeInfo(@TypeOf(layers)) != .array or @typeInfo(@TypeOf(layers)).array.child != LayerType) {
    @compileError("Expected an array of LayerType, got " ++ @typeName(@TypeOf(layers)) ++ " instead");
  }

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      const LOPS = GetLOPS(layers, F, in_training, height, width, .array);
      const LayerOperations = LOPS.LayerOperations;

      const OutputWidth = LayerOperations.Layers[LayerOperations.Layers.len - 1].output_width;
      const OutputHeight = LayerOperations.Layers[LayerOperations.Layers.len - 1].output_height;

      const InputHeight = LayerOperations.getLayerInputDims(0)[0];
      const InputWidth = LayerOperations.getLayerInputDims(0)[1];

      // These can be used as output offsets as we are already given the first input,
      // + we never use the last entry as we are given the last output.
      const CacheSizeArray = LayerOperations.getInputOffsets();
      const CacheSize = CacheSizeArray[CacheSizeArray.len - 1];

      // The size of the largest array that is ever allocated as input/output of any layer
      const MaxLayerSize = std.mem.max(comptime_int, &CacheSizeArray);

      const Retval = struct {
        layers: LayerOperations.LayerInstanceType,
        cache: if (!in_training) void else [CacheSize]F = undefined,

        pub const Gradient = LOPS.Gradient;

        pub fn reset(self: *@This(), rng: std.Random) void {
          inline for (0..LayerOperations.Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (@sizeOf(@TypeOf(@field(self.layers, name))) == 0) {
              // logger.log(&@src(), "Skipped {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
              continue;
            }
            // logger.log(&@src(), "Reset {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
            @field(self.layers, name).reset(rng);
          }
        }

        pub fn forward(
          self: if (in_training) *@This() else *const @This(),
          input: *[InputHeight][InputWidth]F,
          output: *[OutputHeight][OutputWidth]F
        ) void {
          @setEvalBranchQuota(1000_000);

          logger.log(&@src(), "mergein: {d}\n", .{input});
          defer logger.log(&@src(), "mergeout: {d}\n", .{output});

          if (in_training) {
            inline for (0..LayerOperations.Layers.len) |i| {
              const name = std.fmt.comptimePrint("{d}", .{i});
              logger.log(&@src(), "mergelayer({d})in: {d}\n", .{i, if (i == 0) input else @as(
                *[LayerOperations.Layers[i-1].output_height][LayerOperations.Layers[i-1].output_width]F,
                @ptrCast(self.cache[CacheSizeArray[i - 1]..].ptr)
              )});
              @field(self.layers, name).forward(
                if (i == 0) input else @ptrCast(self.cache[CacheSizeArray[i - 1]..].ptr),
                if (i == LayerOperations.Layers.len - 1) output else @ptrCast(self.cache[CacheSizeArray[i]..].ptr),
              );
            }
          } else {
            var buf: [2][MaxLayerSize]F = undefined;
            var p1 = &buf[0];
            var p2 = &buf[1];

            inline for (0..LayerOperations.Layers.len) |i| {
              const name = std.fmt.comptimePrint("{d}", .{i});
              @field(self.layers, name).forward(
                if (i == 0) input else @ptrCast(p1),
                if (i == LayerOperations.Layers.len - 1) output else @ptrCast(p2),
              );
              std.mem.swap(@TypeOf(p1), &p1, &p2);
            }
          }
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [InputHeight][InputWidth]F,
          cache_out: *const [OutputHeight][OutputWidth]F,
          d_prev: *[InputHeight][InputWidth]F,
          d_next: *const [OutputHeight][OutputWidth]F,
          gradient: *Gradient,
          comptime calc_prev: bool,
        ) void {
          @setEvalBranchQuota(1000_000);
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");

          var buf: [2][MaxLayerSize]F = undefined;
          var d1 = &buf[0];
          var d2 = &buf[1];

          inline for (0..LayerOperations.Layers.len) |_i| {
            const i = LayerOperations.Layers.len - 1 - _i;
            const l = LayerOperations.Layers[i];
            const name = std.fmt.comptimePrint("{d}", .{i});
            @field(self.layers, name).backward(
              if (i == 0) cache_in else @ptrCast(self.cache[CacheSizeArray[i-1]..].ptr),
              if (_i == 0) cache_out else @ptrCast(self.cache[CacheSizeArray[i]..].ptr),
              if (i == 0) d_prev else @ptrCast(d1),
              if (_i == 0) d_next else @ptrCast(d2),
              if (l.need_gradient) &@field(gradient.sub, name) else undefined,
              if (i == 0) calc_prev else true,
            );
            std.mem.swap(@TypeOf(d1), &d1, &d2);
          }
        }

        pub fn applyGradient(self: *@This(), gradients: *const Gradient, learning_rate: F) void {
          inline for (0..LayerOperations.Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (@TypeOf(@field(gradients.sub, name)) == void) {
              // logger.log(&@src(), "Skipped: {s}", .{@typeName(@TypeOf(@field(gradients.sub, name)))});
              continue;
            } else {
              // logger.log(&@src(), "Applying Gradients to {s}\n", .{@typeName(@TypeOf(@field(gradients.sub, name)))});
            }
            // logger.writer.print("Applying Gradients to {s}\n", .{@typeName(@TypeOf(@field(gradients.sub, name)))}) catch {};
            @field(self.layers, name).applyGradient(&@field(gradients.sub, name), learning_rate);
          }
        }
      };

      return .{
        .width = OutputWidth,
        .height = OutputHeight,
        .layer = Retval,
      };
    }
  }.getLayer;
}


fn validateAny(T: type) bool {
  if (T == LayerType) return true;

  const layers_typeinfo = @typeInfo(T);
  if (layers_typeinfo == .array) {
    const child_type = layers_typeinfo.array.child;
    if (!validateAny(child_type)) {
      @compileLog("Invalid type `" ++ @typeName(child_type) ++ "` encountered while trying to validate `" ++ @typeName(T) ++ "`");
      return false;
    }
    return true;
  }

  if (layers_typeinfo != .@"struct") {
    @compileLog("Invalid type `" ++ @typeName(T) ++ "` encountered while trying to validate `" ++ @typeName(T) ++ "`, expect LayerType, array or a tuple");
    return false;
  } else if (!layers_typeinfo.@"struct".is_tuple) {
    @compileLog("Invalid type encountered while trying to validate `" ++ @typeName(T) ++ "`, struct `" ++ @typeName(T) ++ "` is not a tuple");
    return false;
  }

  inline for (layers_typeinfo.@"struct".fields) |layer_field| {
    if (!validateAny(layer_field.type)) {
      @compileLog("Invalid type `" ++ @typeName(layer_field.type) ++ "` encountered in field `" ++ layer_field.name ++ "` of `" ++ @typeName(T) ++ "`");
      return false;
    }
  }

  return true;
}

// Merges tuple of layers / layer arrays / nested tuples into a single layer
pub fn mergeAny(layers: anytype) LayerType {
  if (!validateAny(@TypeOf(layers))) {
    @compileError("Layers Type `" ++ @typeName(layers) ++ "` is invalid");
  }

  const T = @TypeOf(layers);
  if (T == LayerType) return layers;

  const layers_typeinfo = @typeInfo(T);
  if (layers_typeinfo == .array) {
    comptime var merged_array: [layers.len]LayerType = undefined;
    inline for (layers, 0..) |l, i| merged_array[i] = mergeAny(l);
    return mergeArray(merged_array);
  }

  const layer_fields = layers_typeinfo.@"struct".fields;

  const getterFn = struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      const LOPS = GetLOPS(init: {
        var gotten: [layer_fields.len]LayerType = undefined;
        for (layer_fields, 0..) |l, i| gotten[i] = mergeAny(@field(layers, l.name));
        break :init gotten;
      }, F, in_training, height, width, .tuple);
      const LayerOperations = LOPS.LayerOperations;

      const CacheSizeArray: [LayerOperations.Layers.len + 1]comptime_int = init: {
        var retval: [LayerOperations.Layers.len + 1]comptime_int = undefined;
        const offsets = LayerOperations.getInputOffsets();
        for (offsets, 0..) |o, i| retval[i] = o;

        const last = LayerOperations.Layers[LayerOperations.Layers.len-1];
        retval[LayerOperations.Layers.len] = retval[LayerOperations.Layers.len-1] + last.output_height * last.output_width;
        break :init retval;
      };

      const OutputHeight = 1;
      const OutputWidth = CacheSizeArray[CacheSizeArray.len - 1];

      const InputHeight = LayerOperations.getLayerInputDims(0)[0];
      const InputWidth = LayerOperations.getLayerInputDims(0)[1];

      const Retval = struct {
        layers: LayerOperations.LayerInstanceType,
        pub const Gradient = LOPS.Gradient;
        // pub const Layers = LayerOperations.Layers;

        pub fn reset(self: *@This(), rng: std.Random) void {
          inline for (0..LayerOperations.Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (@sizeOf(@TypeOf(@field(self.layers, name))) == 0) {
              // logger.log(&@src(), "Skipped {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
              continue;
            }
            // logger.log(&@src(), "Reset {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
            @field(self.layers, name).reset(rng);
          }
        }

        pub fn forward(
          self: if (in_training) *@This() else *const @This(),
          input: *[InputHeight][InputWidth]F,
          output: *[OutputHeight][OutputWidth]F
        ) void {
          @setEvalBranchQuota(1000_000);
          inline for (0..LayerOperations.Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            @field(self.layers, name).forward(input, @ptrCast(output[0][CacheSizeArray[i]..].ptr));
          }
        }

        pub fn backward(
          self: *@This(),
          cache_in: *const [InputHeight][InputWidth]F,
          cache_out: *const [OutputHeight][OutputWidth]F,
          d_prev: *[InputHeight][InputWidth]F,
          d_next: *const [OutputHeight][OutputWidth]F,
          gradient: *Gradient,
          comptime calc_prev: bool,
        ) void {
          @setEvalBranchQuota(1000_000);
          if (!in_training) @compileError("Cant call " ++ @typeName(@This()) ++ ".backward() when not in_training");

          if (calc_prev) {
            for (0..height) |y| {
              for (0..width) |x| {
                d_prev[y][x] = 0;
              }
            }
          }

          var d_buf: if (calc_prev) [InputHeight][InputWidth]F else void = undefined;
          inline for (LayerOperations.Layers, 0..) |l, i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            @field(self.layers, name).backward(
              cache_in, @ptrCast(cache_out[0][CacheSizeArray[i]..].ptr),
              if (calc_prev) &d_buf else undefined, @ptrCast(d_next[0][CacheSizeArray[i]..].ptr),
              if (l.need_gradient) &@field(gradient.sub, name) else .{},
              calc_prev,
            );

            if (calc_prev) {
              for (0..height) |y| {
                for (0..width) |x| {
                  d_prev[y][x] += d_buf[y][x] / @as(F, @floatFromInt(LayerOperations.Layers.len));
                }
              }
            }
          }
        }

        pub fn applyGradient(self: *@This(), gradients: *const Gradient, learning_rate: F) void {
          inline for (0..LayerOperations.Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (@TypeOf(@field(gradients.sub, name)) == void) {
              // logger.log(&@src(), "Skipped: {s}", .{@typeName(@TypeOf(@field(gradients.sub, name)))});
              continue;
            } else {
              // logger.log(&@src(), "Applying Gradients to {s}\n", .{@typeName(@TypeOf(@field(gradients.sub, name)))});
            }
            // logger.writer.print("Applying Gradients to {s}\n", .{@typeName(@TypeOf(@field(gradients.sub, name)))}) catch {};
            @field(self.layers, name).applyGradient(&@field(gradients.sub, name), learning_rate / @as(F, @floatFromInt(LayerOperations.Layers.len)));
          }
        }
      };

      return .{
        .width = OutputWidth,
        .height = OutputHeight,
        .layer = Retval,
      };
    }
  }.getLayer;

  return getterFn;
}


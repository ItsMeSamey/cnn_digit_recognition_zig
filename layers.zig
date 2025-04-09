const std = @import("std");
const logger = @import("logger.zig");

fn CopyPtrAttrs(
  comptime source: type,
  comptime size: std.builtin.Type.Pointer.Size,
  comptime child: type,
) type {
  const info = @typeInfo(source).pointer;
  return @Type(.{
    .pointer = .{
      .size = size,
      .is_const = info.is_const,
      .is_volatile = info.is_volatile,
      .is_allowzero = info.is_allowzero,
      .alignment = info.alignment,
      .address_space = info.address_space,
      .child = child,
      .sentinel_ptr = null,
    },
  });
}

fn AsBytesReturnType(comptime P: type) type {
  const pointer = @typeInfo(P).pointer;
  std.debug.assert(pointer.size == .one);
  const size = @sizeOf(pointer.child);
  return CopyPtrAttrs(P, .one, [size]u8);
}

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
  //   gradient: *Gradient,
  //   comptime calc_prev: bool,
  // ) void
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

      const Gradient = struct {
        filter: [filter_y][filter_x]F,
        bias: F,

        pub fn reset(self: *@This()) void {
          self.bias = 0;
          for (0..filter_y) |i| {
            for (0..filter_x) |j| {
              self[i][j] = 0;
            }
          }
        }

        pub fn add(self: *@This(), other: *const @This()) void {
          self.bias += other.bias;
          for (0..filter_y) |i| {
            for (0..filter_x) |j| {
              self[i][j] += other[i][j];
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
          self.bias = rng.float(F) - 0.5;
          for (0..filter_x) |i| {
            for (0..filter_y) |j| {
              self.filter[i][j] = rng.float(F)*10 - 5;
            }
          }
        }

        pub fn asBytes(self: *const @This()) AsBytesReturnType(*const @This()) {
          return std.mem.asBytes(self);
        }

        pub fn fromBytes(bytes: AsBytesReturnType(*const @This())) @This() {
          return std.mem.bytesAsValue(@This(), bytes).*;
        }

        pub fn forward(self: *const @This(), input: *[height][width]F, output: *[out_height][out_width]F) void {
          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              var sum: F = self.bias;
              for (0..filter_y) |filter_y_offset| {
                for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (in_y >= height and in_x >= width) continue;
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
          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              gradient.bias += d_next[out_y][out_x];
            }
          }

          for (0..height) |y| {
            for (0..width) |x| {
              d_prev[y][x] = 0;
            }
          }

          for (0..out_height) |out_y| {
            for (0..out_width) |out_x| {
              for (0..filter_y) |filter_y_offset| {
                for (0..filter_x) |filter_x_offset| {
                  const in_y = out_y * stride_y + filter_y_offset;
                  const in_x = out_x * stride_x + filter_x_offset;
                  if (in_y >= height and in_x >= width) continue;
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
          self.bias -= learning_rate * gradient.bias;
          for (0..filter_y) |i| {
            for (0..filter_x) |j| {
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
  const LayerFn = getConvolver(2,2,2,2);
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
      comptime var out_width = (width - pool_size_x) / stride_x + 1;
      if (out_width * stride_x < width) out_width += 1;
      comptime var out_height = (height - pool_size_y) / stride_y + 1;
      if (out_height * stride_y < height) out_height += 1;

      const Layer = struct {
        pub const Gradient = void;

        const idxType = std.meta.Int(.unsigned, std.math.log2(pool_size_y*pool_size_x));
        max_idx: if (in_training) [out_height][out_width]idxType else void = if (in_training) undefined else {},

        pub fn init() @This() {
          return .{};
        }

        pub fn reset(_: *@This(), _: std.Random) void {
          // Nothing to reset
        }
        pub fn asBytes(_: *const @This()) null {
          return null; // Nothing to save
        }

        pub fn fromBytes(_: null) @This() {
          return .{}; // Nothing to restore
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
                  if (in_y >= height and in_x >= width) continue;

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

        const as_bytes_len = out_width * (width + 1) * @sizeOf(F);
        pub fn asBytes(self: *const @This()) [as_bytes_len]u8 {
          var retval: [as_bytes_len]u8 = undefined;
          @memcpy(retval[0..width*out_width*@sizeOf(F)], std.mem.asBytes(&self.weights));
          @memcpy(retval[width*out_width*@sizeOf(F)..], std.mem.asBytes(&self.biases));
          return retval;
        }

        pub fn fromBytes(bytes: [as_bytes_len]u8) @This() {
          var self: @This() = undefined;
          @memcpy(std.mem.asBytes(&self.weights), bytes[0..width*out_width*@sizeOf(F)]);
          @memcpy(std.mem.asBytes(&self.biases), bytes[width*out_width*@sizeOf(F)..]);
          return self;
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
              Activate.forward(&output[0], &output[0]);
            } else {
              self.cache_in[i] = sum;
              Activate.forward(&self.cache_in, &output[0]);
            }
          }
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

pub fn mergeArray(layers: anytype) LayerType {
  if (@typeInfo(@TypeOf(layers)) != .array or @typeInfo(@TypeOf(layers)).array.child != LayerType) {
    @compileError("Expected an array of LayerType, got " ++ @typeName(@TypeOf(layers)) ++ " instead");
  }

  const LayerProperties = struct {
    output_height: comptime_int,
    output_width: comptime_int,
    layer_type: type,

    need_gradient: bool,
    gradient_type: type,
    is_simple: bool,

    fn fromLayerType(layer_output: LayerType, F: type, in_training: bool, h_in: comptime_int, w_in: comptime_int) @This() {
      const result = layer_output(F, in_training, h_in, w_in);
      const is_simple = @TypeOf(result.layer.forward) == fn (input: *[h_in][w_in]F) *[result.height][result.width]F;
      const gradient_type = if (is_simple) void else result.layer.Gradient;

      return .{
        .output_height = result.height,
        .output_width = result.width,
        .layer_type = result.layer,
        .need_gradient = !(is_simple or @sizeOf(gradient_type) == 0),
        .gradient_type = gradient_type,
        .is_simple = is_simple,
      };
    }

    fn translateAll(F: type, in_training: bool, h_in: comptime_int, w_in: comptime_int) []const @This() {
      comptime var translated_layers: []const @This() = &.{};
      inline for (layers) |layer| {
        translated_layers = translated_layers ++ &[_]@This(){@This().fromLayerType(
          F,
          layer,
          if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].output_height else h_in,
          if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].output_width else w_in,
          in_training
        )};
      }
      return translated_layers;
    }
  };

  return struct {
    pub fn getLayer(F: type, in_training: bool, height: comptime_int, width: comptime_int) LayerOutputType {
      const LayerOperations = struct {
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

        const Layers = LayerProperties.translateAll(F, in_training, height, width);
        const LayerInstanceType = getType(Layers);

        fn getLayerInputDims(layer_num: comptime_int) [2]comptime_int {
          if (layer_num == 0) return .{height, width};
          const layer = Layers[layer_num - 1];
          return .{layer.output_height, layer.output_width};
        }

        const AsBytesTypes = init: {
          var fields: []const std.builtin.Type.StructField = &.{};
          for (Layers, 0..) |l, i| {
            const byte_returntype = if (@sizeOf(l.layer_type) == 0) void else @typeInfo(l.layer_type.asBytes).@"fn".return_type.?;
            fields = fields ++ &[_]std.builtin.Type.StructField{.{
              .name = std.fmt.comptimePrint("{d}", .{i}),
              .type = byte_returntype,
              .default_value_ptr = null,
              .is_comptime = false,
              .alignment = @alignOf(byte_returntype),
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

        fn getOutputOffsets() [@This().Layers.len]comptime_int {
          comptime var retval: [@This().Layers.len]comptime_int = undefined;
          retval[0] = 0;
          inline for (1..Layers.len) |i| {
            const l = Layers[i];
            retval[i] = retval[i-1] + if (l.is_simple) 0 else l.output_height * l.output_width;
          }
          return retval;
        }
      };

      const GradientsType = struct {
        sub: SubType,

        const SubType = init: {
          var fields: []const std.builtin.Type.StructField = &.{};
          for (LayerOperations.Layers, 0..) |l, i| {
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

        fn reset(self: *@This()) void {
          inline for (LayerOperations.Layers, 0..) |l, i| {
            if (l.need_gradient) @field(self.sub, std.fmt.comptimePrint("{d}", .{i})).reset();
          }
        }

        fn add(self: *@This(), other: *@This()) void {
          inline for (LayerOperations.Layers, 0..) |l, i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (l.need_gradient) @field(self.sub, name).add(&@field(other.sub, name));
          }
        }

        pub fn format(value: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
          _ = fmt;
          _ = options;
          inline for (LayerOperations.Layers, 0..) |l, i| {
            try std.fmt.format(writer, "\n-------- {s} --------\n{any}\n", .{@typeName(l.layer_type), @field(value.sub, std.fmt.comptimePrint("{d}", .{i}))});
          }
        }
      };

      const Retval = struct {
        cache: if (!in_training) void else [CacheSize]F = undefined,

        pub const InputHeight = LayerOperations.getLayerInput(0)[0];
        pub const InputWidth = LayerOperations.getLayerInput(0)[1];

        pub const OutputWidth = LayerOperations.Layers[LayerOperations.Layers.len - 1].output_width;
        pub const OutputHeight = LayerOperations.Layers[LayerOperations.Layers.len - 1].output_height;

        const CacheSizeArray = LayerOperations.getOutputOffsets();
        const CacheSize = CacheSizeArray[CacheSizeArray.len - 1];

        // The size of the largest array that is ever allocated as input/output of any layer
        pub const MaxLayerSize = std.mem.max(comptime_int, &CacheSizeArray);

        pub const Gradient = GradientsType;

        pub fn reset(self: *@This(), rng: std.Random) void {
          inline for (0..@This().Layers.len) |i| {
            const name = std.fmt.comptimePrint("{d}", .{i});
            if (@sizeOf(@TypeOf(@field(self.layers, name))) == 0) {
              // logger.log(&@src(), "Skipped {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
              continue;
            }
            // logger.log(&@src(), "Reset {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
            @field(self.layers, name).reset(rng);
          }
        }

        pub fn asBytes(self: *const @This()) LayerOperations.AsBytesTypes {
          var retval: LayerOperations.AsBytesTypes = undefined;
          inline for (@typeInfo(LayerOperations.Layers).@"struct".fields) |l| {
            if (l.type == void) continue;
            @field(retval, l.name) = @field(self.layers, l.name).asBytes();
          }
          return retval;
        }

        pub fn fromBytes(bytes: LayerOperations.AsBytesTypes) @This() {
          var self: @This() = undefined;
          inline for (@typeInfo(LayerOperations.Layers).@"struct".fields) |l| {
            if (l.type == void) continue;
            @field(self.layers, l.name) = @field(bytes, l.name).fromBytes();
          }
          return self;
        }

        pub fn forward(
          self: if (in_training) *@This() else *const @This(),
          input: *[InputHeight][InputWidth]F,
          output: *[OutputHeight][OutputWidth]F
        ) void {
          @setEvalBranchQuota(1000_000);

          if (in_training) {
            inline for (@This().Layers, 0..) |l, i| {
              const name = std.fmt.comptimePrint("{d}", .{i});
              if (l.is_simple) continue;
              @field(self.layers, name).forward(
                if (i == 0) input else @ptrCast(self.cache[CacheSizeArray[i - 1]..].ptr),
                if (i == LayerOperations.Layers.len - 1) output else @ptrCast(self.cache[CacheSizeArray[i]..].ptr),
              );
            }
          } else {
            var buf: [2][MaxLayerSize]F = undefined;
            var p1 = &buf[0];
            var p2 = &buf[1];

            inline for (LayerOperations.TestingLayers, 0..) |l, i| {
              if (l.is_simple) continue;
              const name = std.fmt.comptimePrint("{d}", .{i});
              @field(self.layers, name).forward(
                if (i == 0) input else p1,
                if (i == LayerOperations.Layers.len - 1) output else p2,
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

          inline for (LayerOperations.Layers, 0..) |l, i| {
            if (l.is_simple) continue;
            @field(self.layers, l.name).backward(
              if (i == 0) cache_in else @ptrCast(self.cache[CacheSizeArray[i - 1]..].ptr),
              if (i == LayerOperations.Layers.len - 1) cache_out else @ptrCast(self.cache[CacheSizeArray[i]..].ptr),
              if (i == 0) d_prev else d1,
              if (i == LayerOperations.Layers.len - 1) d_next else d2,
              if (l.need_gradient) &@field(gradient.sub, l.name) else void,
              if (i == 0) calc_prev else true,
            );
            std.mem.swap(@TypeOf(d1), &d1, &d2);
          }
        }

        pub fn applyGradient(self: *@This(), gradients: *const Gradient, learning_rate: F) void {
          inline for (0..@This().Layers.len) |i| {
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

      std.debug.assert(height == Retval.OutputHeight);
      std.debug.assert(width == Retval.OutputWidth);

      return .{
        .width = Retval.OutputWidth,
        .height = Retval.OutputHeight,
        .layer = Retval,
      };
    }
  }.getLayer;
}


pub fn mergeTuple(layers_tuple: anytype) LayerType {
  const layers_tuple_typeinfo = @typeInfo(@TypeOf(layers_tuple));
  if (layers_tuple_typeinfo != .@"struct" or !layers_tuple_typeinfo.@"struct".is_tuple) {
    @compileError("Expected a tuple of layers");
  }

  const layers_tuple_fields = layers_tuple_typeinfo.@"struct".fields;
  inline for (layers_tuple_fields) |layer_field| {
    if (@typeInfo(layer_field.type) != .array and layer_field.type == [@field(layers_tuple, layer_field.name).len]LayerType) continue;
    @compileError("Field type must be an array of `LayerType`, but " ++ layer_field.name ++ " is " ++ @typeName(layer_field.type));
  }

  @compileError("Not implemented");
}


const std = @import("std");
const logger = @import("logger.zig");
const Layers = @import("layers.zig");

pub fn CNN(
  F: type,
  height: comptime_int, width: comptime_int,
  loss_gen: fn(LEN: comptime_int, F: type) type,
  layers: anytype,
) type {
  @setEvalBranchQuota(1000_000);
  std.debug.assert(@TypeOf(layers) == [layers.len]Layers.LayerType);

  const LayerProperties = struct {
    output_height: comptime_int,
    output_width: comptime_int,
    layer_type: type,

    need_gradient: bool,
    gradient_type: type,
    is_simple: bool,

    fn fromLayerType(layer_output: Layers.LayerType, h_in: comptime_int, w_in: comptime_int, in_training: bool) @This() {
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

    fn translateAll(in_training: bool) []const @This() {
      comptime var translated_layers: []const @This() = &.{};
      inline for (layers) |layer| {
        const h_in = if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].output_height else height;
        const w_in = if (translated_layers.len != 0) translated_layers[translated_layers.len - 1].output_width else width;
        translated_layers = translated_layers ++ &[_]@This(){@This().fromLayerType(layer, h_in, w_in, in_training)};
      }
      return translated_layers;
    }
  };

  const Retval = struct {
    fn getLayerInstanceType(L: []const LayerProperties) type {
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

    pub const OutputWidth = Trainer.Layers[Trainer.Layers.len - 1].output_width;
    const LossFn = loss_gen(OutputWidth, F);

    fn getLayerInputDims(layer_num: comptime_int) [2]comptime_int {
      if (layer_num == 0) return .{height, width};
      const layer = Trainer.Layers[layer_num - 1];
      return .{layer.output_height, layer.output_width};
    }
    fn getLayerInput(layer_num: comptime_int) type {
      const dims = getLayerInputDims(layer_num);
      return [dims[0]][dims[1]]F;
    }

    fn getMaxLayerSize() comptime_int {
      comptime var max = 0;
      inline for (0..Trainer.Layers.len+1) |i| {
        const dims = getLayerInputDims(i);
        const len = dims[0] * dims[1];
        if (len > max) max = len;
      }
      return max;
    }

    // The size of the largest array that is ever allocated as input/output of any layer
    pub const MaxLayerSize = getMaxLayerSize();

    pub const Gradients = struct {
      sub: SubType,

      const SubType = init: {
        var fields: []const std.builtin.Type.StructField = &.{};
        for (Trainer.Layers, 0..) |l, i| {
          fields = fields ++ &[1]std.builtin.Type.StructField{.{
            .name = std.fmt.comptimePrint("{d}", .{i}),
            .type = if (l.need_gradient) l.gradient_type else void,
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(l.layer_type),
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
        inline for (Trainer.Layers, 0..) |l, i| {
          if (l.need_gradient) @field(self.sub, std.fmt.comptimePrint("{d}", .{i})).reset();
        }
      }

      fn add(self: *@This(), other: *@This()) void {
        inline for (Trainer.Layers, 0..) |l, i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (l.need_gradient) @field(self.sub, name).add(&@field(other.sub, name));
        }
      }

      pub fn format(value: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        inline for (Trainer.Layers, 0..) |l, i| {
          try std.fmt.format(writer, "\n-------- {s} --------\n{any}\n", .{@typeName(l.layer_type), @field(value.sub, std.fmt.comptimePrint("{d}", .{i}))});
        }
      }
    };

    pub const Trainer = struct {
      layers: LayerInstanceType,
      pub const Layers = LayerProperties.translateAll(true);
      pub const LayerInstanceType = getLayerInstanceType(@This().Layers);

      pub fn toTester(self: *const @This()) Tester {
        var retval: Tester = undefined;
        inline for (0..layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@This().Layers[i].is_simple) continue;
          @field(retval.layers, name) = @TypeOf(@field(retval.layers, name)).fromBytes(@field(self.layers, name).asBytes());
        }
        return retval;
      }

      fn getCacheSizeArray() [@This().Layers.len + 2]comptime_int {
        comptime var retval: [@This().Layers.len + 2]comptime_int = undefined;
        retval[0] = 0;
        retval[1] = height * width;
        inline for (0..@This().Layers.len) |i| {
          if (@This().Layers[i].is_simple) {
            retval[i+2] = retval[i+1];
            retval[i+1] = retval[i];
          } else {
            const dims = getLayerInputDims(i+1);
            retval[i+2] = retval[i+1] + dims[0] * dims[1];
          }
        }
        return retval;
      }
      const CacheSizeArray = getCacheSizeArray();
      const CacheSize = CacheSizeArray[CacheSizeArray.len - 1];

      // Sets the value in cache
      pub fn forward(self: *@This(), input: [height][width]u8, cache: *[CacheSize]F) void {
        inline for (input, 0..) |row, i| {
          inline for (row, 0..) |val, j| {
            @as(*[height][width]F, @ptrCast(cache))[i][j] = @as(F, @floatFromInt(val));
          }
        }
        @import("read_minst.zig").printImage(@as(*[height][width]F, @ptrCast(cache)));

        inline for (@This().Layers, 0..) |l, i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (l.is_simple) continue;
          // logger.log(&@src(), "Forward input ({s})\n\t{any}\n", .{@typeName(l.layer_type), cache[CacheSizeArray[i]..CacheSizeArray[i+1]]});
          @field(self.layers, name).forward(@ptrCast(cache[if (i == 0) 0 else CacheSizeArray[i]..].ptr), @ptrCast(cache[CacheSizeArray[i+1]..].ptr));
          // logger.log(&@src(), "Forward Output ({s})\n\t{any}\n", .{@typeName(l.layer_type), @as(*[l.output_height*l.output_width]F, @ptrCast(cache[CacheSizeArray[i+1]..].ptr))});
        }
      }

      pub fn backward(self: *@This(), cache: *[CacheSize]F, target: u8, gradients: *Gradients) void {
        var buf: [2][MaxLayerSize]F = undefined;
        var d1 = &buf[0];
        var d2 = &buf[1];
        LossFn.backward(cache[CacheSize - OutputWidth..], target, d2[0..OutputWidth]);
        logger.log(&@src(), "predictions: {any}\n", .{cache[CacheSize - OutputWidth..]});
        logger.log(&@src(), "--- dLoss ({d}) ---\n\t{any}\n", .{target, d2[0..OutputWidth]});

        inline for (0..@This().Layers.len) |_i| {
          const i = @This().Layers.len - 1 - _i;
          const l = @This().Layers[i];
          if (l.is_simple) continue;

          const name = std.fmt.comptimePrint("{d}", .{i});
          @field(self.layers, name).backward(
            @ptrCast(cache[CacheSizeArray[i]..].ptr),
            @ptrCast(cache[CacheSizeArray[i+1]..].ptr),
            @ptrCast(d1),
            @ptrCast(d2),
            &@field(gradients.sub, name),
            i == 0,
          );

          const temp = d1;
          d1 = d2;
          d2 = temp;
        }
      }

      pub fn reset(self: *@This(), rng: std.Random) void {
        const gradients: Gradients = undefined;
        inline for (0..@This().Layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@TypeOf(@field(gradients.sub, name)) == void) continue;
          logger.log(&@src(), "Reset {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
          @field(self.layers, name).reset(rng);
        }
      }

      pub const Options = struct {
        verbose: bool = true,
        batch_size: u32,
        learning_rate: F,
      };
      pub fn train(self: *@This(), iterator_readonly: anytype, options: Options) void {
        var iterator = iterator_readonly;
        var gradients: Gradients = undefined;
        var cache: [CacheSize]F = undefined;

        // inline for (0..@This().Layers.len) |i| {
        //   logger.log(&@src(), "Layer {d}\n", .{i});
        //   logger.log(&@src(), "{any}\n", .{@field(self.layers, std.fmt.comptimePrint("{d}", .{i}))});
        // }

        var step: usize = 0;
        while (true) {
          var i: usize = 0;
          var gross_loss: F = 0;

          step += 1;
          gradients.reset();

          for (0..options.batch_size) |_| {
            const next = iterator.next() orelse break;

            i += 1;
            self.forward(next.image.*, &cache);
            self.backward(&cache, next.label, &gradients);

            const loss = LossFn.forward(cache[CacheSize - OutputWidth..], next.label);
            gross_loss += loss;
            if (options.verbose) {
              logger.writer.print("{d:4}-{d:2} Loss({d}) = {d:.3}\n", .{step, i, next.label, loss*100}) catch {};
              // logger.log(&@src(), "Gradients {any}\n", .{gradients});
            }
          }

          if (!options.verbose) {
            logger.writer.print("Step: {d:4}-{d:2} Loss: {d:.3}\n", .{step, i, gross_loss*100}) catch {};
          }

          self.applyGradients(&gradients, options.learning_rate / @as(F, @floatFromInt(options.batch_size)));

          logger.buffered.flush() catch {};
          if (!iterator.hasNext()) break;
        }
      }

      pub fn applyGradients(self: *@This(), gradients: *Gradients, learning_rate: F) void {
        inline for (0..@This().Layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@TypeOf(@field(gradients.sub, name)) == void) continue;
          // logger.writer.print("Applying Gradients to {s}\n", .{@typeName(@TypeOf(@field(gradients.sub, name)))}) catch {};
          @field(self.layers, name).applyGradient(&@field(gradients.sub, name), learning_rate);
        }
      }
    };

    pub const Tester = struct {
      layers: LayerInstanceType,
      pub const Layers = LayerProperties.translateAll(false);
      pub const LayerInstanceType = getLayerInstanceType(@This().Layers);

      pub fn toTrainer(self: *const @This()) Trainer {
        var retval: Trainer = undefined;
        inline for (0..layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@This().Layers[i].is_simple) continue;
          @field(retval.layers, name) = @TypeOf(@field(retval.layers, name)).fromBytes(@field(self.layers, name).asBytes());
        }
        return retval;
      }

      pub fn forward(self: *const @This(), input: [height][width]u8) [OutputWidth]F {
        var buf: [2][MaxLayerSize]F = undefined;
        var p1 = &buf[0];
        var p2 = &buf[1];

        inline for (input, 0..) |row, i| {
          inline for (row, 0..) |val, j| {
            @as(*[height][width]F, @ptrCast(p1))[i][j] = @as(F, @floatFromInt(val)) / @as(F, 255);
          }
        }

        inline for (@This().Layers, 0..) |l, i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (l.is_simple) continue;
          @field(self.layers, name).forward(@ptrCast(p1), @ptrCast(p2));
          const temp = p1;
          p1 = p2;
          p2 = temp;
        }

        return p1[0..OutputWidth].*;
      }

      pub fn @"test"(self: *const @This(), iterator_ro: anytype) F {
        defer logger.buffered.flush() catch {};
        var retval: usize = 0;
        var n: usize = 0;
        var iterator = iterator_ro;
        while (iterator.next()) |next| {
          n += 1;
          const predictions = self.forward(next.image.*);
          var guess: usize = 0;
          inline for (0..predictions.len) |i| {
            if (predictions[guess] < predictions[i]) guess = i;
          }
          if (guess == next.label) retval += 1;

          logger.writer.print("{d:5} Prediction({d}) = {d}\n", .{n, next.label, guess}) catch {};
        }

        return @as(F, @floatFromInt(retval)) / @as(F, @floatFromInt(n));
      }
    };

    test {
      std.debug.print("\n----------- TRAINING LAYERS -----------\n", .{});
      inline for (Trainer.Layers) |l| std.debug.print("{any}\n", .{l});
      std.debug.print("\n----------- TESTING LAYERS -----------\n", .{});
      inline for (Tester.Layers) |l| std.debug.print("{any}\n", .{l});
    }
  };

  std.debug.assert(Retval.Trainer.Layers[Retval.Trainer.Layers.len - 1].output_height == 1);
  return Retval;
}

test CNN {
  const Layer = @import("layers.zig");
  const Activation = @import("functions_activate.zig");
  const Loss = @import("functions_loss.zig");

  const Iterator = struct {
    rng: std.Random,
    remaining: u32,
    val: [1][3]u8 = undefined,

    fn next(self: *@This()) ?struct {
      image: *[1][3]u8,
      label: u8,
    } {
      if (!self.hasNext()) return null;
      self.remaining -= 1;
      self.val[0][0] = self.rng.intRangeLessThan(u8, 0, 2);
      self.val[0][1] = self.rng.intRangeLessThan(u8, 0, 2);
      self.val[0][2] = self.rng.intRangeLessThan(u8, 0, 2);
      return .{
        .image = &self.val,
        .label = (self.val[0][0] ^ self.val[0][1]) + self.val[0][2],
      };
    }

    pub fn hasNext(self: *const @This()) bool {
      return self.remaining > 0;
    }
  };

  // xor test
  const cnn = CNN(f64, 1, 3, Loss.CategoricalCrossentropy, [_]Layer.LayerType{
    Layer.getDense(9, Activation.Sigmoid),
    Layer.getDense(6, Activation.Sigmoid),
    Layer.getDense(3, Activation.Softmax),
  });

  var trainer = cnn.Trainer{.layers = undefined};

  var rng = std.Random.DefaultPrng.init(@bitCast(std.time.timestamp()));
  trainer.reset(rng.random());

  for (0..4) |i| {
    trainer.train(Iterator{.rng = rng.random(), .remaining = 1024}, .{
      .verbose = true,
      .batch_size = @intCast(i + 32),
      .learning_rate = @as(f64, 0.1) / @as(f64, @floatFromInt(1 + i)),
    });
  }

  var tester = trainer.toTester();
  const accuracy = tester.@"test"(Iterator{.rng = rng.random(), .remaining = 512});
  std.debug.print("\n>> Final Accuracy: {d:.3}%\n", .{accuracy*100});
}


const std = @import("std");
const logger = @import("logger.zig");
const Layers = @import("layers.zig");
const Functions = @import("functions.zig");

pub fn CNN(F: type, height: comptime_int, width: comptime_int, layers: anytype) type {
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
          @field(retval, name) = @TypeOf(@field(retval, name)).fromBytes(@field(self, name).toBytes());
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
          logger.log(&@src(), "Forward Output ({s})\n\t{any}\n", .{@typeName(l.layer_type), @as(*[l.output_height*l.output_width]F, @ptrCast(cache[CacheSizeArray[i+1]..].ptr))});
        }
      }

      const CategoricalCrossentropy = Functions.CategoricalCrossentropy(OutputWidth, F);
      pub fn backward(self: *@This(), cache: *[CacheSize]F, target: u8, gradients: *Gradients) void {
        var buf: [2][MaxLayerSize]F = undefined;
        var d1 = &buf[0];
        var d2 = &buf[1];
        CategoricalCrossentropy.backward(cache[CacheSize - OutputWidth..], target, d2[0..OutputWidth]);
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

      pub const Options = struct {
        verbose: bool = true,
        batch_size: u32,
        learning_rate: F,
        rng: std.Random,
      };
      pub fn train(self: *@This(), iterator_readonly: anytype, options: Options) void {
        var iterator = iterator_readonly;
        var gradients: Gradients = undefined;
        var cache: [CacheSize]F = undefined;

        inline for (0..@This().Layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          logger.log(&@src(), "Reset {s}\n", .{@typeName(@TypeOf(@field(self.layers, name)))});
          if (@TypeOf(@field(gradients.sub, name)) == void) continue;
          @field(self.layers, name).reset(options.rng);
        }

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

            const loss = CategoricalCrossentropy.forward(cache[CacheSize - OutputWidth..], next.label);
            gross_loss += loss;
            if (options.verbose) {
              logger.writer.print("Step: {d:4}-{d:2} Loss: {d:.3}\n", .{step, i, loss*100}) catch {};
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
          @field(retval, name) = @TypeOf(@field(retval, name)).fromBytes(@field(self, name).toBytes());
        }
        return retval;
      }

      pub fn forward(self: *@This(), input: [height][width]u8) [OutputWidth]F {
        var buf: [2][MaxLayerSize]F = undefined;
        var p1 = &buf[0];
        var p2 = &buf[1];

        inline for (input, 0..) |row, i| {
          inline for (row, 0..) |val, j| {
            @as([height][width]F, @ptrCast(p1))[i][j] = @as(F, @floatFromInt(val)) / @as(F, 255);
          }
        }

        inline for (@This().Layers, 0..) |l, i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@typeInfo(l.return_type) == .pointer) continue;
          @field(self.layers, name).forward(@ptrCast(p1), @ptrCast(p2));
          const temp = p1;
          p1 = p2;
          p2 = temp;
        }

        return p1[0..OutputWidth];
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
  _ = comptime CNN(f32, 28, 28, [_]Layers.LayerType{Layers.getFlattener()});
}


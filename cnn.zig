const std = @import("std");
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
    backward_output: type,

    fn fromLayerType(layer_output: Layers.LayerType, h_in: comptime_int, w_in: comptime_int, in_training: bool) @This() {
      const result = layer_output(F, in_training, h_in, w_in);
      const backward_returntype = @typeInfo(@TypeOf(result.layer.backward)).@"fn".return_type.?;

      return .{
        .output_height = result.height,
        .output_width = result.width,
        .layer_type = result.layer,
        .need_gradient = @TypeOf(result.layer.forward) == fn (input: *[height][width]F) *[result.height][result.width]F or @sizeOf(backward_returntype) == 0,
        .backward_output = backward_returntype,
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
            .type = if (l.need_gradient and @typeInfo(l.backward_output) != .pointer) l.backward_output else void,
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
          if (l.need_gradient) @field(self.sub, name).add(@field(other, name));
        }
      }

      fn div(self: *@This(), val: F) void {
        inline for (Trainer.Layers, 0..) |l, i| {
          if (l.need_gradient) @field(self.sub, std.fmt.comptimePrint("{d}", .{i})).div(val);
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

      fn getCacheSizeArray() [@This().Layers.len + 1]comptime_int {
        comptime var retval: [@This().Layers.len + 1]comptime_int = undefined;
        retval[0] = height * width;
        inline for (1..@This().Layers.len+1) |i| {
          if (@typeInfo(@This().Layers[i-1].backward_output) == .pointer) {
            retval[i] = retval[i-1];
          } else {
            const dims = getLayerInputDims(i);
            retval[i] = retval[i-1] + dims[0] * dims[1];
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
            @as(*[height][width]F, @ptrCast(cache))[i][j] = @as(F, @floatFromInt(val)) / @as(F, 255);
          }
        }

        inline for (@This().Layers, 0..) |l, i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@typeInfo(l.backward_output) == .pointer) continue;
          @field(self.layers, name).forward(@ptrCast(cache[CacheSizeArray[i]..].ptr), @ptrCast(cache[CacheSizeArray[i+1]..].ptr));
        }
      }

      const CategoricalCrossentropy = Functions.CategoricalCrossentropy(OutputWidth, F);
      pub fn backward(self: *@This(), cache: *[CacheSize]F, target: u8) Gradients {
        var buf: [2][MaxLayerSize]F = undefined;
        var d1 = &buf[0];
        var d2 = &buf[1];
        CategoricalCrossentropy.backward(cache[CacheSize - OutputWidth..], target, d1[0..OutputWidth]);

        var gradients: Gradients = undefined;

        inline for (0..@This().Layers.len) |_i| {
          const i = @This().Layers.len - 1 - _i;
          const l = @This().Layers[i];
          if (@typeInfo(l.backward_output) == .pointer) continue;
          const name = std.fmt.comptimePrint("{d}", .{i});
          const grad_out = @field(self.layers, name).backward(
            @ptrCast(&cache[CacheSizeArray[i]..]),
            @ptrCast(&cache[CacheSizeArray[i+1]..]),
            @ptrCast(d1),
            @ptrCast(d2),
          );
          if (l.need_gradient) @field(gradients.sub, name) = grad_out;

          const temp = d1;
          d1 = d2;
          d2 = temp;
        }

        return gradients;
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

        var step: usize = 0;
        while (true) {
          step += 1;
          var i: usize = 1;
          const n = iterator.next() orelse break;
          self.forward(n.image.*, &cache);
          gradients = self.backward(&cache, n.label);

          for (1..options.batch_size) |_| {
            const next = iterator.next() orelse break;
            self.forward(next.image.*, &cache);
            var grad = self.backward(&cache, next.label);
            gradients.add(&grad);
            i += 1;

            const loss = CategoricalCrossentropy.forward(cache[CacheSize - OutputWidth..], next.label);
            if (options.verbose) std.debug.print("Step:{d:4} Loss: {d}\n", .{step, loss});
          }

          gradients.div(@floatFromInt(i));
          self.applyGradients(&gradients, options.learning_rate);
          if (!iterator.hasNext()) break;
        }
      }

      pub fn applyGradients(self: *@This(), gradients: *Gradients, learning_rate: F) void {
        inline for (0..@This().Layers.len) |i| {
          const name = std.fmt.comptimePrint("{d}", .{i});
          if (@TypeOf(@field(gradients.sub, name)) == void) continue;
          @field(self.layers, name).applyGradient(@field(gradients.sub, name), learning_rate);
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

